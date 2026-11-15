# free-model-aggregator-entry

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/OWNER/REPO/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/OWNER/REPO/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free-model aggregator server in one Python file (standard library only). It reads the trial brief, probes ten simulated free-tier providers in their own wires (OpenAI, Anthropic, Gemini), registers one catalog, then serves OpenAI Chat Completions and Anthropic Messages requests -- JSON and SSE -- always routing to the highest-rated free model that meets each request's requirements, with same-provider retry and failover when a provider refuses.

## Run it

```sh
python3 main.py          # inside a trial: MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE set
python3 main.py --list   # print the catalog document and exit
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=REFERRAL_CODE&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
