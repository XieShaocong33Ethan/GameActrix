import openai
import re

# Module 1: Self Reflection System Prompt
SELF_REFLECTION_SYSTEM_PROMPT = """
You are an AI assistant that assesses the progress of playing Super Mario and provides useful guidance.

GAME RULES
- *Reach the Flag*: Navigate through the level and reach the flagpole before time runs out
- *Avoid Enemies*: Defeat or bypass enemies using jumps or power-ups
- *Collect Power-ups*: Gain abilities by collecting mushrooms and flowers. When powered-up, collisions with enemies reduce size instead of causing death
- *Preserve Lives*: Avoid hazards such as pits and enemies to stay alive

Object Descriptions
- Bricks: Breakable blocks; may contain items or coins (Size: 16x16)
- Question Blocks: Reveal coins or power-ups when hit; deactivate after use (Size: 16x16)
- Pit: Falling in results in losing a life
- Warp Pipe: Raised above the ground, so Mario must jump over them when it appear in front (Size: 30xHeight(y))
- Monster Goomba: Basic enemy; can be defeated by jumping on it (Size: 16x16)
- Monster Koopa: Turtle enemy; retreats into shell when jumped on (Size: 20x24)
- Item Mushroom: Grows Mario larger, grants protection (Size: 16x16)
- Stairs: Used to ascend/descend terrain
- Flag: Touch to complete the level
- Ground: the ground level in the game is y=32

Action Descriptions
- Mario (Size: 13x13) continuously moves to the right at a fixed speed
- You must choose an appropriate jump level to respond to upcoming obstacles
- Each jump level determines both:
    - How far Mario jumps horizontally (x distance)
    - How high Mario reaches at the peak of the jump (y height)
- Jump Levels *(values based on flat ground jumps)*:
    - Level 0: +0 in x, +0 in y (No jump, just walk)
    - Level 1: +42 in x, +35 in y
    - Level 2: +56 in x, +46 in y
    - Level 3: +63 in x, +53 in y
    - Level 4: +70 in x, +60 in y
    - Level 5: +77 in x, +65 in y
    - Level 6: +84 in x, +68 in y
    - *Note*: The values above assume Mario is jumping from flat ground. When jumping from elevated platforms or interacting with mid-air obstacles (e.g., bricks), the actual jump trajectory and landing position may vary.
- The key is choosing the *right jump level at the right moment*
- *Use higher levels* to jump over taller or farther obstacles
- Consider *the size* of Mario and objects
- While jumping, Mario follows a *parabolic arc*, moving upward and then downward in a smooth curve, so Mario can be *blocked by objects mid-air or be defeated by airborne enemies*
- Mario can step on top of bricks, blocks, warp pipes, and stairs

The game state is expressed in the following format:
Position of Mario: (x, y)
Position of all objects:
- Bricks: [(x1, y1), (x2, y2), ...]
- Question Blocks: [(x1, y1), ...]
- Inactivated Blocks: [(x1, y1), ...]
- Monster Goombas: [(x1, y1), ...]
- Monster Koopas: [(x1, y1), ...]
- Pit: start at (x1, y1), end at (x2, y2)
- Warp Pipes: [(x1, y1, height), ...]
- Item Mushrooms: [(x1, y1), ...]
- Stair Blocks: [(x1, y1), ...]
- Flag: (x, y)
(Note: All (x, y) positions refer to the top-left corner of each object)


You will receive the past game state, Mario's past action, and the current game state
Your job is to evaluate Mario's past actions and provide critiques for improving his action or avoiding potential dangers


You should only respond in the format as described below:
### Critique
[Describe critique here]


EXAMPLES

Input:
### Past Game State
Position of Mario: (100, 40)
Positions of all objects:
- Blocks: (112,40), (112,56), (112, 72), (112, 88)
- Monster Goombas: (180,40)

### Past Mario Action
Jump Level 4

### Current Game State
Position of Mario: (100, 40)
Positions of all objects:
- Blocks: (112,40), (112,56), (112, 72), (112, 88)
- Monster Goombas: (160,40)

Output:
### Critique
Mario is blocked by a high wall blocks. Jump with Level 6 to get past it.


Input:
### Past Game State
Position of Mario: (122, 40)
Positions of all objects:
- Warp Pipes: (200,100)
- Monster Goombas: (180,40)

### Past Mario Action
Jump Level 0

### Current Game State
Position of Mario: (122, 40)
Positions of all objects:
- Warp Pipes: (190,100)
- Monster Goombas: (160,40)

Output:
### Critique
Mario didn't jump since there were no obstacles, which is a reasonable move. Stay alert for approaching Monster Goombas.
"""

SELF_REFLECTION_USER_PROMPT = """### Past Game State
{prev_state_str}

### Past Mario Action
{past_mario_action}

### Current Game State
{cur_state_str}
"""

# Module 2: Subtask Planning System Prompt
SUBTASK_PLANNING_SYSTEM_PROMPT = """You are an AI assistant for the Super Mario game. Your role is to plan long-term actions for Mario based on the given game state.

GAME RULES
- *Reach the Flag*: Navigate through the level and reach the flagpole before time runs out
- *Avoid Enemies*: Defeat or bypass enemies using jumps or power-ups
- *Collect Power-ups*: Gain abilities by collecting mushrooms and flowers. When powered-up, collisions with enemies reduce size instead of causing death
- *Preserve Lives*: Avoid hazards such as pits and enemies to stay alive

Object Descriptions
- Bricks: Breakable blocks; may contain items or coins (Size: 16x16)
- Question Blocks: Reveal coins or power-ups when hit; deactivate after use (Size: 16x16)
- Pit: Falling in results in losing a life
- Warp Pipe: Raised above the ground, so Mario must jump over them when it appear in front (Size: 30xHeight(y))
- Monster Goomba: Basic enemy; can be defeated by jumping on it (Size: 16x16)
- Monster Koopa: Turtle enemy; retreats into shell when jumped on (Size: 20x24)
- Item Mushroom: Grows Mario larger, grants protection (Size: 16x16)
- Stairs: Used to ascend/descend terrain
- Flag: Touch to complete the level
- Ground: the ground level in the game is y=32

Action Descriptions
- Mario (Size: 15x13) continuously moves to the right at a fixed speed
- You must choose an appropriate jump level to respond to upcoming obstacles
- Each jump level determines both:
    - How far Mario jumps horizontally (x distance)
    - How high Mario reaches at the peak of the jump (y height)
- Jump Levels *(values based on flat ground jumps)*:
    - Level 0: +0 in x, +0 in y (No jump, just walk)
    - Level 1: +42 in x, +35 in y
    - Level 2: +56 in x, +46 in y
    - Level 3: +63 in x, +53 in y
    - Level 4: +70 in x, +60 in y
    - Level 5: +77 in x, +65 in y
    - Level 6: +84 in x, +68 in y
    - *Note*: The values above assume Mario is jumping from flat ground. When jumping from elevated platforms or interacting with mid-air obstacles (e.g., bricks), the actual jump trajectory and landing position may vary.
- The key is choosing the *right jump level at the right moment*
- *Use higher levels* to jump over taller or farther obstacles
- Consider *the size* of Mario and objects
- While jumping, Mario follows a *parabolic arc*, moving upward and then downward in a smooth curve, so Mario can be *blocked by objects mid-air or be defeated by airborne enemies*
- Mario can step on top of bricks, blocks, warp pipes, and stairs

The game state is expressed in the following format:
Position of Mario: (x, y)
Position of all objects:
- Bricks: [(x1, y1), (x2, y2), ...]
- Question Blocks: [(x1, y1), ...]
- Inactivated Blocks: [(x1, y1), ...]
- Monster Goombas: [(x1, y1), ...]
- Monster Koopas: [(x1, y1), ...]
- Pit: start at (x1, y1), end at (x2, y2)
- Warp Pipes: [(x1, y1, height), ...]
- Item Mushrooms: [(x1, y1), ...]
- Stair Blocks: [(x1, y1), ...]
- Flag: (x, y)
(Note: All (x, y) positions refer to the top-left corner of each object)


You will receive the current game state for the Mario's last action.
Your job is to analyze the state, and figure out a safe and efficient long-term path forward, avoiding obstacles whenever possible.
You should respond precise, consie, and strategic descriptions of *cations* and *subtask (plan)* to do so.

You MUST only respond in the format as below:
### Cautions
[List any dangers or things to avoid here]

### Subtask
[Describe specific intermediate plans or steps Mario should take to reach the flagpole]


EXAMPLE

Input:
### Game State
Position of Mario: (100, 40)
Positions:
- Bricks: (140,90), (170,90), (200, 130), (230, 130)
- Monster Goombas: (150, 40), (170, 40), (190, 40)

Output:
### Cautions
A group of Goomba monsters is approaching, so be cautious with your jump level while on the ground.

### Subtask
To safely bypass the Goombas, use the bricks in the air as stepping platforms.
"""

SUBTASK_PLANNING_USER_PROMPT = """### Game State
{cur_state_str}

### Last Action Critique
{critique}
"""

# Module 3: Action Inference System Prompt
ACTION_INFERENCE_SYSTEM_PROMPT = """You are an AI assistant playing the Super Mario game. Your goal is to reach the flagpole at the end of each level without dying by avoiding obstacles, collecting power-ups, and defeating/avoiding enemies

GAME RULES
- *Reach the Flag*: Navigate through the level and reach the flagpole before time runs out
- *Avoid Enemies*: Defeat or bypass enemies using jumps or power-ups
- *Collect Power-ups*: Gain abilities by collecting mushrooms and flowers. When powered-up, collisions with enemies reduce size instead of causing death
- *Preserve Lives*: Avoid hazards such as pits and enemies to stay alive

Object Descriptions
- Bricks: Breakable blocks; may contain items or coins (Size: 16x16)
- Question Blocks: Reveal coins or power-ups when hit; deactivate after use (Size: 16x16)
- Pit: Falling in results in losing a life
- Warp Pipe: Raised above the ground, so Mario must jump over them when it appear in front (Size: 30xHeight(y))
- Monster Goomba: Basic enemy; can be defeated by jumping on it (Size: 16x16)
- Monster Koopa: Turtle enemy; retreats into shell when jumped on (Size: 20x24)
- Item Mushroom: Grows Mario larger, grants protection (Size: 16x16)
- Stairs: Used to ascend/descend terrain
- Flag: Touch to complete the level
- Ground: the ground level in the game is y=32

Action Descriptions
- Mario (Size: 15x13) continuously moves to the right at a fixed speed
- You must choose an appropriate jump level to respond to upcoming obstacles
- Each jump level determines both:
    - How far Mario jumps horizontally (x distance)
    - How high Mario reaches at the peak of the jump (y height)
- Jump Levels *(values based on flat ground jumps)*:
    - Level 0: +0 in x, +0 in y (No jump, just walk)
    - Level 1: +42 in x, +35 in y
    - Level 2: +56 in x, +46 in y
    - Level 3: +63 in x, +53 in y
    - Level 4: +70 in x, +60 in y
    - Level 5: +77 in x, +65 in y
    - Level 6: +84 in x, +68 in y
    - *Note*: The values above assume Mario is jumping from flat ground. When jumping from elevated platforms or interacting with mid-air obstacles (e.g., bricks), the actual jump trajectory and landing position may vary.
- The key is choosing the *right jump level at the right moment*
- *Use higher levels* to jump over taller or farther obstacles
- Consider *the size* of Mario and objects
- While jumping, Mario follows a *parabolic arc*, moving upward and then downward in a smooth curve, so Mario can be *blocked by objects mid-air or be defeated by airborne enemies*
- Mario can step on top of bricks, blocks, warp pipes, and stairs


At each game step, you will receive the current game state in the following format:
Position of Mario: (x, y)
Position of all objects:
- Bricks: [(x1, y1), (x2, y2), ...]
- Question Blocks: [(x1, y1), ...]
- Inactivated Blocks: [(x1, y1), ...]
- Monster Goombas: [(x1, y1), ...]
- Monster Koopas: [(x1, y1), ...]
- Pit: start at (x1, y1), end at (x2, y2)
- Warp Pipes: [(x1, y1, height), ...]
- Item Mushrooms: [(x1, y1), ...]
- Stair Blocks: [(x1, y1), ...]
- Flag: (x, y)
(Note: All (x, y) positions refer to the top-left corner of each object)

You should then respond with
Explain (if applicable): Why you choose the jump level
Jump Level: n (where n is an integer from 0 to 6, indicating the chosen jump level)

You MUST only respond in the format with the prefix '### Actions\n' as below:

### Actions
Explain: ...
Jump Level: n


EXAMPLE

Input:
### Game State
Position of Mario: (100, 40)
Positions of all objects:
- Monster Goombas: (140, 40)

Output:
### Actions
Explain: Goomba is 40 units ahead on flat ground — a level 1 jump (+42 X) is sufficient to jump over or stomp it.
Jump Level: 1
"""

ACTION_INFERENCE_USER_PROMPT = """
### Last Action Critique
{critique}

### Mid-term Playing Strategy
{subtask_description}

### Game State
{cur_state_str}
"""

MODEL = "gpt-5-nano"

class OpenAIMarioAgent:
    TRACK = "TRACK1"

    def __init__(self):
        self.client = openai.OpenAI()
        
        # Memory tracking
        self.step_count = 0
        self.prev_state_str = None
        self.last_action = None
    
    def _parse_section(self, text, section_name):
        """Extract content after a markdown section header"""
        pattern = rf"### {section_name}\s*(.+?)(?=###|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _module_self_reflection(self, prev_state_str, cur_state_str, past_action):
        """Module 1: Self Reflection - Critique past action"""
        print("\n=== MODULE 1: Self Reflection ===")
        
        user_prompt = SELF_REFLECTION_USER_PROMPT.format(
            prev_state_str=prev_state_str,
            past_mario_action=past_action,
            cur_state_str=cur_state_str
        )
        
        response = self.client.responses.create(
            model=MODEL,
            input=user_prompt,
            instructions=SELF_REFLECTION_SYSTEM_PROMPT,
            reasoning={"effort": "low"}
        )
        
        output = response.output_text.strip()
        print(f"Self Reflection Output:\n{output}\n")
        
        # Parse the critique
        critique = self._parse_section(output, "Critique")
        if not critique:
            critique = "No specific critique for the last action."
        
        return critique
    
    def _module_subtask_planning(self, cur_state_str, critique):
        """Module 2: Subtask Planning - Plan mid-term strategy"""
        print("\n=== MODULE 2: Subtask Planning ===")
        
        user_prompt = SUBTASK_PLANNING_USER_PROMPT.format(
            cur_state_str=cur_state_str,
            critique=critique
        )
        
        response = self.client.responses.create(
            model=MODEL,
            input=user_prompt,
            instructions=SUBTASK_PLANNING_SYSTEM_PROMPT,
            reasoning={"effort": "low"}
        )
        
        output = response.output_text.strip()
        print(f"Subtask Planning Output:\n{output}\n")
        
        # Parse cautions and subtask
        cautions = self._parse_section(output, "Cautions")
        subtask = self._parse_section(output, "Subtask")
        
        if not cautions:
            cautions = "No specific cautions."
        if not subtask:
            subtask = "Continue moving forward toward the flagpole."
        
        # Combine for subtask description
        subtask_description = f"Cautions: {cautions}\nSubtask: {subtask}"
        
        return subtask_description
    
    def _module_action_inference(self, cur_state_str, critique, subtask_description):
        """Module 3: Action Inference - Choose immediate action"""
        print("\n=== MODULE 3: Action Inference ===")
        
        user_prompt = ACTION_INFERENCE_USER_PROMPT.format(
            critique=critique,
            subtask_description=subtask_description,
            cur_state_str=cur_state_str
        )
        
        response = self.client.responses.create(
            model=MODEL,
            input=user_prompt,
            instructions=ACTION_INFERENCE_SYSTEM_PROMPT,
            reasoning={"effort": "low"}
        )
        
        output = response.output_text.strip()
        print(f"Action Inference Output:\n{output}\n")
        
        return output
    
    def act(self, obs):
        """
        Act loop: all 3 modules in sequence
        1. Self Reflection (skip on first step)
        2. Subtask Planning
        3. Action Inference
        """
        print("\n" + "="*60)
        print(f"STEP {self.step_count + 1}: Starting Reflection-Planning Agent Pipeline")
        print("="*60)
        
        cur_state_str = obs.get("obs_str", "")
        
        # MODULE 1: Self Reflection
        if self.step_count > 0 and self.prev_state_str and self.last_action:
            critique = self._module_self_reflection(
                self.prev_state_str,
                cur_state_str,
                self.last_action
            )
        else:
            critique = "N/A (first step)"
            print("\n=== MODULE 1: Self Reflection ===")
            print("Skipped (first step)\n")
        
        # MODULE 2: Subtask Planning
        subtask_description = self._module_subtask_planning(cur_state_str, critique)
        
        # MODULE 3: Action Inference
        action_output = self._module_action_inference(
            cur_state_str,
            critique,
            subtask_description
        )
        
        # Parse action from Module 3 output
        match = re.search(r"Jump Level:\s*(\d+)", action_output, re.IGNORECASE)
        
        if match:
            jump_level = match.group(1)
            action = f"Jump Level: {jump_level}"
        else:
            action = "Jump Level: 0"
        
        self.step_count += 1
        self.prev_state_str = cur_state_str
        self.last_action = action
        
        print("\n" + "="*60)
        print(f"✓ FINAL ACTION: {action}")
        print("="*60 + "\n")
        
        return action
