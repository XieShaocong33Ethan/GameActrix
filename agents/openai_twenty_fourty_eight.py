import openai
import re

SYSTEM_PROMPT = """
You are an expert AI agent specialized in playing the 2048 game with advanced strategic reasoning. 
Your primary goal is to achieve the highest possible tile value while maintaining long-term playability by preserving the flexibility of the board and avoiding premature game over. 

### 2048 Game Rules ### 
1. The game is played on a 4×4 grid. Tiles slide in one of four directions: 'up', 'down', 'left', or 'right'. 
2. Only two **consecutive tiles** with the SAME value can merge. Merges cannot occur across empty tiles. 
3. **Merging is directional**: 
   - Row-based merges occur on 'left' or 'right' actions. 
   - Column-based merges occur on 'up' or 'down' actions. 
4. **All tiles first slide in the chosen direction as far as possible**, then merges are applied. 
5. **A tile can merge only once per move**. When multiple same-value tiles are aligned (e.g., [2, 2, 2, 2]), merges proceed from the movement direction. For example: 
   - [2, 2, 2, 2] with 'left' results in [4, 4, 0, 0]. 
   - [2, 2, 2, 0] with 'left' results in [4, 2, 0, 0]. 
6. An action is only valid if it causes at least one tile to slide or merge. Otherwise, the action is ignored, and no new tile is spawned. 
7. After every valid action, a new tile (usually **90 percent chance of 2, 10 percent chance of 4**) appears in a random empty cell. 
8. The game ends when the board is full and no valid merges are possible. 
9. Score increases only when merges occur, and the increase equals the value of the new tile created from the merge. 

### Decision Output Format ### 
Analyze the provided game state and determine the **single most optimal action** to take next. 
Return your decision in the following exact format: 
### Reasoning
<a detailed summary of why this action was chosen>
### Actions
<up, right, left, or down>

Ensure that: 
- The '### Reasoning' field provides a clear explanation of why the action is the best choice, including analysis of current tile positions, merge opportunities, and future flexibility. 
- The '### Actions' field contains only one of the four valid directions. 
"""

USER_PROMPT = """
### Target task
{task_description}

### Previous state
{prev_state_str}

### Last executed action
{action}

### Current state
{cur_state_str}

You should only respond in the format described below, and you should not output comments or other information.
Provide your response in the strict format: 
### Reasoning
<a detailed summary of why this action was chosen>
### Actions
<direction>
"""

MODEL = "gpt-5-nano"

class OpenAITwentyFourtyEightAgent:
    TRACK = "TRACK1"
    
    def __init__(self):
        self.client = openai.OpenAI()
        self.prev_state_str = "N/A"
        self.last_action = "No action yet"

    def act(self, obs):
        game_info = obs.get("game_info", {})
        cur_state_str = obs.get("obs_str", "")

        response = self.client.responses.create(
            model=MODEL,
            input=USER_PROMPT.format(task_description=game_info.get("task_description", ""),
                            prev_state_str=self.prev_state_str, 
                            action=self.last_action, 
                            cur_state_str=cur_state_str),
            instructions=SYSTEM_PROMPT,
            reasoning={
                "effort": "low",
            }
        )
        action = self._parse_actions(response.output_text.strip())
        if action not in ["left", "right", "up", "down"]:
            action = "left" # Fall back to left if the action is not valid

        self.prev_state_str = cur_state_str
        self.last_action = action

        return action

    def _parse_actions(self, output):
        """
        Return the full string after ### Actions.
        """
        actions_match = re.search(r"### Actions\s*\n(.+)", output, re.IGNORECASE | re.DOTALL)
        if actions_match:
            actions_section = actions_match.group(1).strip()
            return actions_section
        return ""