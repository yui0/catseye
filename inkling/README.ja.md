# Cat's eyeで動かすInkling 2 Mini

[English](README.md)

Thinking MachinesのInkling 2 Miniを、手元のCPUで動かせます。端末でのテキスト生成や対話に加え、OpenAI互換APIを通じてOpenCodeなどのクライアントからも使えます。付属の変換ツールは、量子化したテキスト用の重み、トークナイザー、モデル設定を1つのsafetensorsファイルにまとめます。モデルの詳細と重みのライセンスは、[Inkling 2 Miniのモデルカード](https://huggingface.co/thinkingmachines/Inkling-2-Mini)を参照してください。

## ビルド

Cat's eyeのチェックアウトのルートから`inkling/`に移動してビルドします。ビルドには`make`とC11対応コンパイラが必要です。`inkling`と`inkling_quantize`が生成されます。

```sh
cd inkling
make
```

macOSで複数のスレッドを使うには、ビルド前に`brew install libomp`でOpenMPを導入します。1スレッドで動かす場合は`make OPENMP=0`でビルドできます。

## モデルの準備

`inkling_quantize`で変換済みのファイルがあれば、この手順を省き、そのファイルのパスを`-m`で指定してください。元のチェックポイントから変換する場合は、[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)で必要なファイルを取得します。

```sh
hf download thinkingmachines/Inkling-2-Mini \
  model-00001-of-00001.safetensors config.json tokenizer.json \
  --local-dir Inkling-2-Mini

./inkling_quantize -i Inkling-2-Mini -o inkling-q4.safetensors
```

既定の変換結果は約15.5 GBです。元のチェックポイントと変換後のファイルを保存できる空き容量を確保してください。変換ツールは結果を書き出すまでメモリに保持するため、変換にはRAMが32 GB程度あるマシンを使うと余裕があります。

## 端末から使う

質問を1件送る例、対話を続ける例、UTF-8のテキストファイルからプロンプトを読む例です。

```sh
./inkling -m inkling-q4.safetensors -p '空が青い理由を一文で説明して。'
./inkling -m inkling-q4.safetensors -i
./inkling -m inkling-q4.safetensors -f prompt.txt --effort low -n 256 -v
```

対話モードでは終了まで会話がKVキャッシュに残ります。`Ctrl+D`で終了できます。`-s`でシステムメッセージを指定し、`--think`でモデルの思考内容を表示できます。`-n`は生成トークン数の上限で、既定は512です。`-c`はコンテキスト長で、既定は4096です。思考を短めにするには`--effort low`、処理時間を表示するには`-v`を使います。

## OpenCodeやAPIクライアントから使う

`inkling/`でサーバーを起動します。この例のコンテキスト長と出力上限は、付属のOpenCode 2向け設定に合わせています。

```sh
OMP_WAIT_POLICY=PASSIVE ./inkling -m inkling-q4.safetensors \
  --serve 8080 -c 16384 -n 2048 --effort low -v
```

サーバーの待受先は`127.0.0.1`です。推論リクエストは1件ずつ処理します。コンテキスト長を増やすとメモリ使用量も増えます。別の端末で接続を確認し、短い質問を送ってみてください。

```sh
curl -fsS http://127.0.0.1:8080/v1/models

curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling","messages":[{"role":"user","content":"一言で挨拶して。"}],"max_tokens":128}'
```

チャットAPIは、テキスト入力、JSON応答、`"stream":true`を指定した場合のSSE配信、思考テキスト、関数呼び出しとツール結果の履歴に対応します。ツールの選択方法は`auto`と`none`を指定できます。

OpenCode 2では、付属の[opencode.json](opencode.json)を使います。OpenCodeで作業するプロジェクトに移動し、手元のCat's eyeチェックアウトにある設定ファイルの絶対パスを`OPENCODE_CONFIG`に指定してください。

```sh
OPENCODE_CONFIG=/path/to/catseye/inkling/opencode.json \
  opencode run --standalone -m catseye/inkling 'このプロジェクトの構成を説明して'
```

サーバーのポートやコンテキスト長を変えた場合は、設定ファイルの`settings.baseURL`や`limit.context`も合わせて変更してください。APIに渡すモデルIDは`inkling`です。[OpenCodeのカスタムプロバイダー設定](https://opencode.ai/docs/models/#compatible)

## メモリが少ない環境で使う

エキスパートをQ3、アテンションと語彙テーブルをQ4にすると、モデルファイルは約11.9 GBになります。メモリに余裕のあるマシンで変換してから、推論を行うマシンへファイルをコピーできます。

```sh
./inkling_quantize -i Inkling-2-Mini -o inkling-q3.safetensors -b q3 -a q4 -e q4
./inkling -m inkling-q3.safetensors -c 2048 --batch 32 -p '水の循環を説明して。' -v
```

Inklingはモデルファイルをメモリマップし、推論に使うエキスパートを必要に応じて読み込みます。エキスパートの重みがメモリに収まりきらない場合は、自動でエキスパートキャッシュを使います。読み込みが頻繁に発生する環境では、SSDの速度が生成速度に影響します。まずは既定のエキスパート数とループ回数で試してください。ストレージからの読み込みに時間がかかる場合は、`--topk 6`で読み込む量を減らせますが、回答の品質に影響することがあります。

## テスト

`make test`はSIMDカーネルを検証し、小さなモデルを生成して推論結果をPyTorchの参照実装と比較します。実行にはPython、PyTorch、NumPy、`safetensors`が必要です。`make test-tokenizer MODEL_DIR=Inkling-2-Mini`は、取得したトークナイザーファイルを使ってHugging Faceの`tokenizers`と照合します。
