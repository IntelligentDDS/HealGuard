# 扫描器上游源码

`bandit/` 已随本包提供，请勿重新克隆覆盖；其他源码和工具不提供二进制或规则快照，以下命令在发行目录根路径执行，仅适用于对应目录尚不存在时。

```bash
git clone https://github.com/github/codeql.git refs/codeql
git -C refs/codeql checkout 14450f5bf38ea9a3ce2e0e45dcf51c0bbdd01af5
git clone https://github.com/semgrep/semgrep-rules.git refs/semgrep-rules
git -C refs/semgrep-rules checkout e5b5a42ec061854378c11e0d01f19250b52bc2e9
git clone https://github.com/facebook/pyre-check.git refs/pyre-check
git -C refs/pyre-check checkout d5614e0a75f539443225fce3c3c55a77cfbfb72c
uv run python scripts/run_build_sink_rules.py
```

恢复规则前核对上游许可；规则构建需要这些源码目录，不只是安装扫描器的 pip 包；默认 Sink 扫描为 Bandit、Semgrep、CodeQL，Pysa 和 Joern 不在默认工具列表中。

CodeQL CLI 固定参考版本为 **2.26.1**，上游下载地址：`https://github.com/github/codeql-cli-binaries/releases/download/v2.26.1/codeql-linux64.zip`，可解压到 `.tools/` 形成 `.tools/codeql/codeql`，也可配置 `PATH`；安装后以 `codeql version` 或 `.tools/codeql/codeql version` 检查。

Semgrep 命令需另行安装，当前包不锁定其 CLI 版本，因此跨环境扫描一致性尚未验证；所有上游来源与已包含代理的版本见 `docs/THIRD_PARTY.md`。
