Applied the bilingual docs policy across the whole docs/ folder and pushed it.

Added an EN→ZH switch link ([简体中文](...)) near the top of every English doc under docs/.
Created missing Simplified Chinese companions for all existing English docs (e.g. docs/tools-index.zh-CN.md), and ensured every ZH doc has a back-link (English).
ZH files are structured as “中文导读（章节列表）+ English 原文” to keep commands/logs/code symbols 100% faithful while still providing a Chinese entry point.
Pushed in commit deca839 on origin/main.

If you want, I can now go back and replace the “中文导读” sections with full translations for the top N most important docs (starting from the shorter ones).
