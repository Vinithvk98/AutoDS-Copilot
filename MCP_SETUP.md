# Connect AutoDS to Claude Desktop (MCP)

This exposes the AutoDS pipeline as tools an AI client can call. Once connected,
you can say things like "profile data/sample_customers.csv and train the best
model for churn" and Claude will run the pipeline for you.

## 1. Install the requirements

Make sure the Python you will point Claude at has the packages installed.

```
cd ~/Desktop/VINI_1/AutoDS_Copilot
python3 -m pip install -r requirements.txt
```

Find the exact Python you just used, you will need its full path in a moment:

```
which python3
```

Copy that path (for example `/usr/bin/python3` or a venv path).

## 2. Open the Claude Desktop config

The file lives here on macOS:

```
~/Library/Application Support/Claude/claude_desktop_config.json
```

If it does not exist, create it. In Claude Desktop you can also open it from
Settings, then Developer, then Edit Config.

## 3. Add the AutoDS server

Paste this in, merging with anything already there. Replace `python3` with the
full path from step 1 if `python3` alone does not work.

```json
{
  "mcpServers": {
    "autods": {
      "command": "python3",
      "args": ["/Users/prakruthiprakash/Desktop/VINI_1/AutoDS_Copilot/mcp_launch.py"]
    }
  }
}
```

## 4. Restart Claude Desktop

Quit it fully and reopen. You should see AutoDS listed under the tools icon,
with eight tools available.

## 5. Try it

Ask Claude something like:

- "Use autods to profile sample_customers.csv"
- "Run the full AutoDS pipeline on sample_reviews.csv with target sentiment"
- "Ask autods for guidance on handling imbalanced classes"

Bundled sample names like `sample_customers.csv` resolve automatically. For your
own files, give a full path such as `/Users/you/Downloads/mydata.csv`.

## The eight tools

profile_dataset, detect_task, run_eda, recommend_cleaning, model_leaderboard,
train_and_evaluate, full_run, guidance.

## Troubleshooting

- Nothing shows up: the command path is wrong. Use the full path to the Python
  that has the requirements installed, and the full path to `mcp_launch.py`.
- A tool errors on a file: pass an absolute path to your dataset.
- Check the Claude Desktop logs under `~/Library/Logs/Claude` if a server fails
  to start.
