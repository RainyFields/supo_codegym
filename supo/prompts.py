"""Prompts used by the SUPO/GRPO CodeGym agent loop.

The task system/user prompts come verbatim from the CodeGym dataset
(VanishD/CodeGym, task_en_instruction_en_env). Only the summarization
instruction (paper Appendix B.2, CodeGym variant) and the continuation
template (paper Appendix C.1.1 rollout example: "We are in the following stage
of solving the problem: ...") are defined here.
"""

# Appendix B.2 of the paper, CodeGym summarization prompt v_sum. The paper shows it
# as a "System:" turn; Qwen3.5's chat template forbids mid-conversation system
# messages, so we send it as a user turn with identical content.
SUMMARY_PROMPT = (
    "You are a helpful agent interacting with a function calling environment to solve user's problem. "
    "The interaction history is now too long. Please summarize the interaction history.\n"
    "- Remember to keep the important information in the history to ensure that you can continue "
    "solving the problem.\n"
    "- Do not call any function in this turn.\n"
    "Now generate the summary, and put your summary inside tag <summary></summary>."
)

# Appendix C.1.1: the next trajectory starts from the original prompt followed by the
# summary of the previous trajectory.
CONTINUATION_TEMPLATE = (
    "{user_prompt}\n\n"
    "We are in the following stage of solving the problem:\n{summary}"
)

FC_BEGIN = "<|FunctionCallBegin|>"
FC_END = "<|FunctionCallEnd|>"

NO_CALL_OBSERVATION = (
    "No function call was detected in your response. Please call exactly one of the provided "
    f"functions, wrapped as {FC_BEGIN}[{{\"name\": ..., \"parameters\": {{...}}}}]{FC_END}."
)
