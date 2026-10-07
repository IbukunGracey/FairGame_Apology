# FAIRGAME_Apology: Evaluating LLM Apology Behaviors in Game-Theoretic Contexts

FAIRGAME_Apology is an extension of the original FAIRGAME framework, specifically designed to investigate and evaluate how Large Language Models (LLMs) exhibit apology behaviors in strategic interaction scenarios. This project builds upon the core FAIRGAME game-theoretic framework to systematically study LLM behavioral patterns when apologizing, particularly in response to different game outcomes.

## Overview

This extension introduces controlled game phases triggered by specific payoff outcomes from the Prisoner's Dilemma-style interactions:

- **R (Reward)**: Both agents cooperate - mutual cooperation outcome
- **S (Sucker's payoff)**: Focal agent cooperates while opponent defects - exploitation scenario
- **T (Temptation)**: Focal agent defects while opponent cooperates - temptation to defect
- **P (Punishment)**: Both agents defect - mutual defection outcome

After each outcome, the framework evaluates how LLMs generate and structure apologies, enabling researchers to:
- Analyze apology patterns across different strategic contexts
- Compare apology behaviors between different LLM models
- Examine the impact of memory on apology generation and strategic adaptation

## Memory Configurations

The framework supports multiple memory configurations to study how context retention affects apology behaviors:

- **No Memory**: Agents respond without previous interaction history
- **Memory 1**: Agents retain the last interaction
- **Extended Memory**: Agents maintain longer interaction histories (configurable)

These memory variations allow researchers to investigate whether and how LLMs adjust their apology strategies based on accumulated interaction context.

## Apology-Decision-Action Workflow

In controlled mode, the framework implements a systematic interaction sequence where game outcomes are predetermined rather than played. The outcomes follow after sequence R, S , T and P, allowing researchers to capture apology behavior and its impact on subsequent strategic choices:

1. **Controlled Outcome Phase**: The game outcome (R, S, T, or P) is set by the framework in a controlled sequence, rather than through actual gameplay

2. **Apology Decision Phase**: Based on the controlled outcome, agents decide whether to apologize
   - If apologizing: The agent provides its apology message along with the reasoning for why it chose to apologize
   - If not apologizing: The agent proceeds without an apology

3. **Next Action Phase**: After seeing the opponent's apology decision and message (if provided), each agent makes their next game choice (cooperate or defect)

This controlled workflow enables researchers to:
- Systematically examine apology behaviors across all payoff outcomes in a standardized sequence
- Investigate whether and how apologies influence subsequent cooperation or defection decisions
- Analyze the relationship between apology types/content and behavioral changes
- Identify patterns in how different LLMs respond to apologies from other agents
- Study the strategic use of apologies in controlled game-theoretic scenarios
- Investigate the authenticity and effectiveness of LLM-generated apologies in rebuilding cooperation

## Code Repository Structure

The following tree shows the list of the repository's sections and their main contents:

```
└── apy.py             # Flask API for local testing and interaction
└── Dockerfile         # Containerization setup
└── main.py            # Entry point script to run the core application. It also provides an example of the input
└── resources/         # Static resources (JSON config files and templates)
└── results/           # Stores output results, logs, or evaluation metrics
└── src/               # Core source code: models, logic, and processing pipelines
└── unit_tests/        # Unit tests to verify the functionality of components
```

## Requirements

Your project needs the following keys in the .env file (an example is provided in .env.example):

- API_KEY_OPENAI to properly connect to OpenAI's API and models.
- API_KEY_MISTRAL to properly connect to Mistral's API and models.
- API_KEY_ANTHROPIC to properly connect to Anthropic's API and models.

Optionally, to enable saving results to an S3-compatible storage, you can also include:
- S3_ENDPOINT
- S3_KEY
- S3_SECRET
- BUCKET_NAME

## About This Research

This project is part of doctoral research at Teesside University, investigating how Large Language Models exhibit and utilize apology behaviors in strategic interaction contexts.


## License

[[License: Apache License Version 2.0]](http://www.apache.org/licenses/)

The source code for the site is licensed under the Apache License Version 2.0, which you can find in the LICENSE file.