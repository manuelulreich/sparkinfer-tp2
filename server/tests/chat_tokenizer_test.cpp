#include "chat_tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <utility>

namespace {

#define CHECK(expr)                                                                            \
    do {                                                                                       \
        if (!(expr)) {                                                                         \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr);             \
            return false;                                                                      \
        }                                                                                      \
    } while (0)

bool test_thinking_prompt_and_nonstream_parser() {
    sparkinfer_server::ChatRequest request;
    sparkinfer_server::ChatMessage message;
    message.role = "user";
    message.content = "Explain briefly.";
    request.messages.push_back(std::move(message));
    const std::string prompt = sparkinfer_server::apply_qwen36_tools_template(request, true);
    const std::string suffix = "<|im_start|>assistant\n<think>\n";
    CHECK(prompt.size() >= suffix.size());
    CHECK(prompt.compare(prompt.size() - suffix.size(), suffix.size(), suffix) == 0);

    const auto parsed = sparkinfer_server::parse_assistant_output(
        "private reasoning\n</think>\n\nPublic answer.<|im_end|>", true, false);
    CHECK(parsed.error.empty());
    CHECK(parsed.reasoning_content == "private reasoning");
    CHECK(parsed.content == "Public answer.");
    CHECK(parsed.tool_calls.empty());

    // Defensively tolerate a model that repeats the already-prefilled opening marker.
    const auto repeated = sparkinfer_server::parse_assistant_output(
        "<think>\nreason\n</think>\nanswer", true, false);
    CHECK(repeated.reasoning_content == "reason");
    CHECK(repeated.content == "answer");

    // Same tolerance when the repeated marker isn't the very first byte (e.g. a stray leading
    // token before it) -- the marker itself must never leak into reasoning_content.
    const auto repeated_not_at_start = sparkinfer_server::parse_assistant_output(
        " <think>\nreason\n</think>\nanswer", true, false);
    CHECK(repeated_not_at_start.reasoning_content == "reason");
    CHECK(repeated_not_at_start.content == "answer");
    CHECK(repeated_not_at_start.reasoning_content.find("<think>") == std::string::npos);
    return true;
}

bool test_thinking_stream_boundaries() {
    sparkinfer_server::ThinkingStreamSplitter splitter(true, false);
    std::string reasoning;
    std::string content;
    for (const char* piece : {"private ", "reasoning\n</thi", "nk>\n\nPublic ",
                              "answer.<|im_", "end|>"}) {
        const auto delta = splitter.feed(piece);
        reasoning += delta.reasoning_content;
        content += delta.content;
    }
    sparkinfer_server::ThinkingStreamSplitter::Delta tail;
    splitter.finish(tail);
    reasoning += tail.reasoning_content;
    content += tail.content;
    CHECK(reasoning == "private reasoning\n");
    CHECK(content == "Public answer.");
    CHECK(content.find("<think>") == std::string::npos);
    CHECK(content.find("</think>") == std::string::npos);
    CHECK(content.find("<|im_end|>") == std::string::npos);
    return true;
}

bool test_thinking_stream_repeated_opening_marker() {
    // Streaming starts directly in the "inside <think>" phase (the prompt already primed
    // "<think>\n"), but must still tolerate -- and strip -- a defensively repeated opening
    // marker the same way the non-streaming parser does, rather than leaking it into
    // reasoning_content.
    sparkinfer_server::ThinkingStreamSplitter splitter(true, false);
    std::string reasoning;
    std::string content;
    for (const char* piece : {"<thi", "nk>\nreason\n</thi", "nk>\nanswer"}) {
        const auto delta = splitter.feed(piece);
        reasoning += delta.reasoning_content;
        content += delta.content;
    }
    sparkinfer_server::ThinkingStreamSplitter::Delta tail;
    splitter.finish(tail);
    reasoning += tail.reasoning_content;
    content += tail.content;
    // Streaming reasoning_content chunks aren't leading-whitespace-trimmed the way the
    // non-streaming parser's final string is (pre-existing, orthogonal to this fix) -- the
    // marker itself must simply never appear in the output.
    CHECK(reasoning == "\nreason\n");
    CHECK(content == "answer");
    CHECK(reasoning.find("<think>") == std::string::npos);
    return true;
}

bool test_nonthinking_stream_hides_terminal_marker() {
    sparkinfer_server::ThinkingStreamSplitter splitter(false, false);
    std::string content;
    for (const char* piece : {"hello", "<|im_", "end|>"})
        content += splitter.feed(piece).content;
    sparkinfer_server::ThinkingStreamSplitter::Delta tail;
    splitter.finish(tail);
    content += tail.content;
    CHECK(content == "hello");
    return true;
}

bool test_stop_filter_full_match_single_chunk() {
    sparkinfer_server::StopSequenceFilter filter({"STOP"});
    const std::string safe = filter.feed("hello STOP world");
    CHECK(filter.matched());
    CHECK(safe == "hello ");
    return true;
}

bool test_stop_filter_match_split_across_chunks() {
    sparkinfer_server::StopSequenceFilter filter({"STOP"});
    std::string safe = filter.feed("hello ST");
    CHECK(!filter.matched());
    CHECK(safe == "hello ");  // "ST" held back as a possible partial prefix of "STOP"
    safe += filter.feed("OP more");
    CHECK(filter.matched());
    CHECK(safe == "hello ");
    return true;
}

bool test_stop_filter_multiple_stops_earliest_wins() {
    sparkinfer_server::StopSequenceFilter filter({"world", "STOP"});
    const std::string safe = filter.feed("hello STOP world");
    CHECK(filter.matched());
    CHECK(safe == "hello ");  // "STOP" occurs first even though it's second in the list
    return true;
}

bool test_stop_filter_prefix_of_another_stop() {
    // Shorter stop fires as soon as it completes, regardless of list order -- matches real
    // OpenAI/vLLM/llama.cpp behavior, not a bug.
    sparkinfer_server::StopSequenceFilter filter({"a", "ab"});
    const std::string safe = filter.feed("xy a z");
    CHECK(filter.matched());
    CHECK(safe == "xy ");
    return true;
}

bool test_stop_filter_no_match_never_swallows_permanently() {
    sparkinfer_server::StopSequenceFilter filter({"NEVER_APPEARS"});
    std::string safe;
    for (const char* piece : {"hello ", "there ", "world"}) safe += filter.feed(piece);
    CHECK(!filter.matched());
    CHECK(safe == "hello there world");
    return true;
}

bool test_stop_filter_empty_stop_list_is_passthrough() {
    sparkinfer_server::StopSequenceFilter filter({});
    const std::string safe = filter.feed("anything at all");
    CHECK(!filter.matched());
    CHECK(safe == "anything at all");
    return true;
}

bool test_stop_filter_marker_with_embedded_nul() {
    // A JSON string can encode an embedded NUL byte; std::string preserves it but strlen()
    // would silently truncate there. Indirectly exercises the marker_prefix_len(std::string)
    // overload StopSequenceFilter relies on for holdback -- if it truncated at the NUL, this
    // stop string would falsely match on "A" alone instead of requiring the full "A\0B".
    const std::string stop_with_nul = std::string("A\0B", 3);
    sparkinfer_server::StopSequenceFilter filter({stop_with_nul});
    std::string safe = filter.feed(std::string("xA", 2));
    CHECK(!filter.matched());  // "A" alone must not be treated as a full match
    safe += filter.feed(std::string("\0By", 3));
    CHECK(filter.matched());
    CHECK(safe == "x");
    return true;
}

bool test_muse_rejects_all_tool_protocol_history() {
    sparkinfer_server::ChatRequest parsed;
    std::string error;
    const std::string request = R"JSON({
      "messages":[
        {"role":"assistant","content":null,"tool_calls":[{
          "id":"call_1","type":"function",
          "function":{"name":"lookup","arguments":"{\"query\":\"x\"}"}
        }]},
        {"role":"tool","tool_call_id":"call_1","content":"ok"},
        {"role":"user","content":"continue"}
      ],
      "tools":[{"type":"function","function":{
        "name":"lookup",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}}}
      }}],
      "tool_choice":"none"
    })JSON";
    CHECK(sparkinfer_server::parse_chat_request_json(request, parsed, error));
    CHECK(!sparkinfer_server::validate_chat_request_model_support(parsed, true, error));
    CHECK(error.find("supported only for Qwen3.6") != std::string::npos);
    error.clear();
    CHECK(sparkinfer_server::validate_chat_request_model_support(parsed, false, error));
    return true;
}

bool test_tool_choice_none_still_uses_strict_output_parser() {
    const std::string request_json = R"JSON({
      "messages":[{"role":"user","content":"Do not call tools."}],
      "tools":[{"type":"function","function":{
        "name":"terminal",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}}}
      }}],
      "tool_choice":"none"
    })JSON";
    sparkinfer_server::ChatRequest request;
    std::string error;
    CHECK(sparkinfer_server::parse_chat_request_json(request_json, request, error));
    const auto forbidden = sparkinfer_server::parse_assistant_output(
        "<tool_call>\n<function=terminal>\n<parameter=command>\nid\n</parameter>\n"
        "</function>\n</tool_call>",
        false, false, &request);
    CHECK(!forbidden.error.empty());
    CHECK(forbidden.content.empty());
    CHECK(forbidden.tool_calls.empty());

    const auto answer = sparkinfer_server::parse_assistant_output(
        "No tool used.<|im_end|>", false, false, &request);
    CHECK(answer.error.empty());
    CHECK(answer.content == "No tool used.");
    return true;
}

bool test_openrouter_reasoning_enablement() {
    CHECK(sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"enabled":true}})", false));
    CHECK(!sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"enabled":false,"effort":"high"}})", true));
    CHECK(sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"effort":"low"}})", false));
    CHECK(sparkinfer_server::parse_enable_thinking(
        R"({"reasoning_effort":"medium"})", false));
    CHECK(!sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"effort":"none"}})", true));
    CHECK(sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"max_tokens":512}})", false));
    return true;
}

}  // namespace

// GPT-2/ByteLevel-BPE byte-table round-trip -- pure function, no loaded tokenizer needed. "Ġ" is
// U+0120 (UTF-8 0xC4 0xA0), the standard byte-level-BPE stand-in for the space byte 0x20.
bool test_gpt2_bytelevel_decode_space_marker() {
    const auto piece = sparkinfer_server::gpt2_bytelevel_decode("\xC4\xA0hello");
    CHECK(!piece.bytes.empty());
    CHECK(piece.bytes[0] == 0x20);
    CHECK(piece.display == " hello");
    return true;
}

// U+0122 (UTF-8 0xC4 0xA2) is the byte-level-BPE stand-in for raw byte 0x80 -- a lone UTF-8
// continuation byte, not valid standalone text. display must show a replacement character
// (matches real OpenAI's behavior for tokens that split a multi-byte character).
bool test_gpt2_bytelevel_decode_invalid_utf8_shows_replacement() {
    const auto piece = sparkinfer_server::gpt2_bytelevel_decode("\xC4\xA2");
    CHECK(piece.bytes.size() == 1 && piece.bytes[0] == 0x80);
    CHECK(piece.display.find("\xEF\xBF\xBD") != std::string::npos);
    return true;
}

bool test_gpt2_bytelevel_decode_empty_piece() {
    const auto piece = sparkinfer_server::gpt2_bytelevel_decode("");
    CHECK(piece.bytes.empty());
    CHECK(piece.display.empty());
    return true;
}

// Printable ASCII maps to itself under the byte-level table -- a plain word round-trips exactly.
bool test_gpt2_bytelevel_decode_printable_ascii_roundtrip() {
    const auto piece = sparkinfer_server::gpt2_bytelevel_decode("hello");
    CHECK(piece.bytes.size() == 5);
    CHECK(piece.display == "hello");
    return true;
}

// decode_delta decodes a window of the last tokens. Streamed through it, a text must come back
// delta for delta as the whole-output decode gives it, and concatenated exactly as decode() of
// all its tokens (no U+FFFD for characters split over tokens) (needs a tokenizer.json:
// SPARKINFER_TEST_TOKENIZER=<path>; skipped without one).
bool test_decode_delta_window_matches_full() {
    const char* path = std::getenv("SPARKINFER_TEST_TOKENIZER");
    if (!path) { std::printf("decode_delta window: skipped (no SPARKINFER_TEST_TOKENIZER)\n"); return true; }
    sparkinfer_server::ChatTokenizer tok;
    std::string err;
    CHECK(tok.load(path, err));
    const std::string texts[] = {
        "Plain ASCII, with  double  spaces,\ttabs and\nnewlines.\n\n```cpp\nint main() { return 0; }\n```",
        "Gr\xc3\xbc\xc3\x9f Gott \xe2\x80\x94 \xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81 "
        "\xf0\x9f\x98\x80\xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd \xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa7 "
        "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87 "
        "\xeb\x8c\x80\xed\x95\x9c\xeb\xaf\xbc\xea\xb5\xad \xe2\x88\x91\xe2\x88\xab\xe2\x88\x9a",
        "<think>\nreasoning\n</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\n/a/b\n</parameter>\n</function>\n</tool_call>",
    };
    for (const std::string& base : texts) {
        std::string text;
        for (int r = 0; r < 6; r++) text += base;   // longer than the window many times over
        const std::vector<int> ids = tok.encode_raw(text);
        CHECK(ids.size() > 64);
        std::vector<int> acc, all;
        std::string streamed;
        for (int id : ids) {
            const std::string got = tok.decode_delta(acc, id);
            // The whole-output form decode_delta replaced (with the same hold-back of an
            // incomplete trailing character).
            all.push_back(id);
            auto strip = [](std::string t) {
                while (t.size() >= 3 && t.compare(t.size() - 3, 3, "\xEF\xBF\xBD") == 0) t.resize(t.size() - 3);
                return t;
            };
            const std::string full = strip(tok.decode(all));
            std::string want = full;
            if (all.size() > 1) {
                const std::string prev = strip(tok.decode(std::vector<int>(all.begin(), all.end() - 1)));
                size_t i = 0;
                const size_t n = std::min(full.size(), prev.size());
                while (i < n && full[i] == prev[i]) ++i;
                while (i > 0 && i < full.size() && (static_cast<unsigned char>(full[i]) & 0xC0) == 0x80) --i;
                want = full.substr(i);
            }
            CHECK(got == want);
            streamed += got;
        }
        CHECK(streamed == tok.decode(ids));
    }
    return true;
}

int main() {
    if (!test_thinking_prompt_and_nonstream_parser()) return 1;
    if (!test_thinking_stream_boundaries()) return 1;
    if (!test_thinking_stream_repeated_opening_marker()) return 1;
    if (!test_nonthinking_stream_hides_terminal_marker()) return 1;
    if (!test_stop_filter_full_match_single_chunk()) return 1;
    if (!test_stop_filter_match_split_across_chunks()) return 1;
    if (!test_stop_filter_multiple_stops_earliest_wins()) return 1;
    if (!test_stop_filter_prefix_of_another_stop()) return 1;
    if (!test_stop_filter_no_match_never_swallows_permanently()) return 1;
    if (!test_stop_filter_empty_stop_list_is_passthrough()) return 1;
    if (!test_stop_filter_marker_with_embedded_nul()) return 1;
    if (!test_muse_rejects_all_tool_protocol_history()) return 1;
    if (!test_tool_choice_none_still_uses_strict_output_parser()) return 1;
    if (!test_openrouter_reasoning_enablement()) return 1;
    if (!test_gpt2_bytelevel_decode_space_marker()) return 1;
    if (!test_gpt2_bytelevel_decode_invalid_utf8_shows_replacement()) return 1;
    if (!test_gpt2_bytelevel_decode_empty_piece()) return 1;
    if (!test_gpt2_bytelevel_decode_printable_ascii_roundtrip()) return 1;
    if (!test_decode_delta_window_matches_full()) return 1;
    std::printf("chat_tokenizer_test: OK\n");
    return 0;
}
