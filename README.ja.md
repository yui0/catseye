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

`make help`で全体のコマンドを表示します。`make test`でライブラリのテスト(`test/`:全レイヤーの数値微分による勾配チェック、バッチ正規化・データ拡張・safetensors・量子化・GEMMの検証、小規模な学習の通し確認)を実行します。文字起こしは`make -C transcript`でビルドし、`transcript/`内で`./cats`を実行します。モデルの標準配置先は`transcript/models/`です。詳しい使い方は[文字起こしのREADME](transcript/README.ja.md)を参照してください。

従来のルート直下の学習サンプルは`make legacy`でビルドできます。ライブラリのAPIとサンプルは[英語のREADME](README.md#features)を参照してください。

## 学習率とバッチサイズ

全レイヤー(全結合・畳み込み・逆畳み込み・バッチ正規化)がバッチ平均の勾配を使い、バイアスも重みと同じオプティマイザで更新します。そのため学習率`eta`はどのレイヤーでも同じ意味になります。以前は畳み込みだけが平均、他は合計でした。以前の設定を保つには、全結合・逆畳み込みレイヤーの`eta`にバッチサイズを掛けてください。
