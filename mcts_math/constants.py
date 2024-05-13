TIMEOUT_SECONDS = 30
TIMEOUT_MESSAGE = f"Execution of the code snippet has timed out for exceeding {TIMEOUT_SECONDS} seconds."


QUESTION_COLOR = "light_red"
PARTIAL_SOL_COLOR = "green"
SOLUTION_COLOR = "light_green"
OBSERVATION_COLOR = "light_yellow"
ERROR_COLOR = "red"
WARNING_COLOR = "light_magenta"


TOO_MANY_CODE_ERRORS = "Too many consecutive steps have code errors."
TOO_MANY_STEPS = "Fail to sove the problem within limited steps."
NO_VALID_CHILD = "Fail to generate parsable text for next step."


FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)

OBSERVATION = "Observation: "
THOUGHT = "Thought: "
ACTION_INPUT = "Action Input: "
ACTION = "Action: "

CODE_LTAG = "<code>"
CODE_RTAG = "</code>"
OBSERVATION_LTAG = "<output>"
OBSERVATION_RTAG = "</output>"
STEP_LTAG = "<step>"
STEP_RTAG = "</step>"
PARA_LTAG = "<p>"
PARA_RTAG = "</p>"