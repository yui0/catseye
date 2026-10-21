# Cat's Eye

Cで実装したニューラルネットライブラリと、その応用プログラムです。

[English](README.md)

| 場所 | 内容 |
|---|---|
| `catseye.h` | 共通のニューラルネットライブラリ |
| `catseye_llm.h` | 共通のGGUF・LLM推論ランタイム |
| [`transcript/`](transcript/README.ja.md) | 文字起こしCLI、音声・ONNXヘッダー、モデル変換ツール、Colab、テスト |
| [`llm/`](llm/README.md) | 従来のLLMサンプルとGPU実装 |
| `example/`、`super_resolution/`、`word2vec/` | ニューラルネットのサンプル |
| `tools/` | ライブラリ全体のツール |

`make help`で全体のコマンドを表示します。文字起こしは`make -C transcript`でビルドし、`transcript/`内で`./cats`を実行します。モデルの標準配置先は`transcript/models/`です。詳しい使い方は[文字起こしのREADME](transcript/README.ja.md)を参照してください。

従来のルート直下の学習サンプルは`make legacy`でビルドできます。ライブラリのAPIとサンプルは[英語のREADME](README.md#features)を参照してください。
