# 🤖 Implementing Your Custom Agent

This guide will walk you through the process of creating your own agent and integrating it into the evaluation framework.

## 1. Agent Structure

Every agent you create must follow a specific structure to be compatible with the evaluation environment. At its core, an agent is a Python class that contains two essential methods: `__init__` and `act`.

-   **`__init__(self)`**: The constructor for your agent. This is where you can initialize any variables, models, or clients that your agent will need throughout the game. For example, you might initialize an OpenAI client here, as seen in `openai_twenty_fourty_eight.py`.

-   **`act(self, obs)`**: This is the most critical method of your agent. It is called at every step of the game, and its role is to decide on the next action to take based on the current game state.

### The `obs` Dictionary

The `act` method receives a single argument, `obs`, which is a dictionary containing all the information your agent needs to make a decision. Here's a breakdown of its contents:

-   `obs['obs_str']` (str): A string representation of the current game state. This is the primary textual observation your agent will use.
-   `obs['obs_image_str']` (str): A base64 encoded string of a PIL image representing the current visual state of the game. You can decode this to get a visual representation of the game, which is crucial for vision-based agents.
-   `obs['game_info']` (dict): A dictionary containing additional contextual information.
    -   `game_info['prev_state_str']` (str or None): The string representation of the previous game state. This can be useful for understanding the immediate consequences of the last action.
    -   `game_info['task_description']` (str): A description of the primary objective for the current game.

## 2. The Action Space

The `act` method must return a single action that is valid for the current game's environment. The action space varies from game to game. For instance, the "2048" game accepts one of the following strings: `"left"`, `"right"`, `"up"`, or `"down"`.

For a detailed list of the specific action spaces for each game, please refer to the environment documentation at `docs/envs.md`.

## 3. Boilerplate Agent

Here is a simple boilerplate template you can use as a starting point for your custom agent:

```python
class MyCustomAgent:
    def __init__(self):
        """
        Initialize your agent here.
        This could include loading models, setting up API clients, etc.
        """
        print("MyCustomAgent initialized.")

    def act(self, obs):
        """
        This method is called at every step of the game.
        
        Args:
            obs (dict): A dictionary containing the game observation.
        
        Returns:
            action: The action to be taken by the agent.
        """
        
        # --- Your agent's logic goes here ---
        
        # Example: Accessing parts of the observation
        current_state_text = obs['obs_str']
        task_description = obs['game_info']['task_description']
        
        print("Current State:", current_state_text)
        print("Task:", task_description)
        
        # For demonstration, we'll just return a random action for 2048.
        # Replace this with your sophisticated decision-making logic!
        import random
        possible_actions = ["left", "right", "up", "down"]
        action = random.choice(possible_actions)
        
        return action
```

## 4. Registering Your Agent

Once you've created your custom agent, you need to tell the evaluation system to use it for the corresponding game. This is done in the `agents/config.py` file.

1.  **Import your agent class**: Add an import statement at the top of `agents/config.py` to import your new agent class.
2.  **Assign your agent**: The file contains variables for each game (e.g., `TwentyFourtyEightAgent`, `SuperMarioAgent`). Assign your custom agent class to the variable for the game you want to compete in.

For example, to use `MyCustomAgent` for the 2048 game, you would modify `agents/config.py` like this:

```python
# agents/config.py

# ... other imports
from agents.my_custom_agent import MyCustomAgent # Step 1: Import your agent

# ... other agent assignments
TwentyFourtyEightAgent = MyCustomAgent # Step 2: Assign your agent
# ...
```

## 5. Advanced Example

For a more advanced example, take a look at `openai_twenty_fourty_eight.py`. This agent demonstrates how to:
-   Use an external library (`openai`).
-   Structure system and user prompts for an LLM.
-   Parse the model's response to extract a valid action.
-   Include retry logic for handling invalid outputs.

We are incredibly excited to see the innovative agents you develop. Good luck!
