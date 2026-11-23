# Inkling-2-Mini (Cat's eye)

Thinking Machines の [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)(総 26.4B、1トークンあたりアクティブ 3.4B、ループ3回で実効 5.4B)を、C(依存ライブラリなし)で動かすテキスト専用のCPU推論です。20GBのチェックポイントを **1ファイル** に圧縮する変換ツール付きです。

モデル固有の部分だけが`inkling/`にあり、汎用部品は`../catseye.h`に入れました:
FP8 / E8M0 / FP4 と NVFP4 / MXFP8 のデコード、`cats_softcap`、`cats_l2_clip`、byte-level BPEトークナイザ(`cats_bpe`。GPT系のpre-tokenizer正規表現と`tokenizer.json`変換を含む)、小さなJSONヘルパー。既存のsafetensors読み込み、Q4_0/Q8_0 GEMM、RMSNormをそのまま使います。

## 使い方

```sh
make                                              # inkling と inkling_quantize
hf download thinkingmachines/Inkling-2-Mini --local-dir Inkling-2-Mini
./inkling_quantize -i Inkling-2-Mini -o inkling-q4.safetensors      # 約15.5GB・1ファイル(元は20GB)

./inkling -m inkling-q4.safetensors -p "空が青い理由を一文で説明して。"
./inkling -m inkling-q4.safetensors -i                 # 対話。会話はKVキャッシュに残る
./inkling -m inkling-q4.safetensors -f prompt.txt --effort low -n 256 -v
```

`--effort`(none/minimal/low/medium/high/xhigh/max または0.99までの数値)、`--think`(思考も表示)、`--raw`(制御トークンも表示)、`--passes N`(ループ回数を減らして高速化・品質は低下)、`-c`(コンテキスト長、既定4096)。

## 8GBのMac(Apple M2)で動かす

標準のQ4ファイル(15.5GB)は8GBのRAMに載らないので、小さい量子化とSSDからのストリーミングを使います:

```sh
# 32GBほどRAMのあるマシンで変換(結果をメモリに保持する)。できたファイルをMacへコピー
./inkling_quantize -i Inkling-2-Mini -o inkling-q3.safetensors -b q3 -a q4 -e q4        # 約11.9GB

# Mac側
brew install libomp                    # Apple clang用のOpenMP。無いとシングルスレッドになる
make && make test                      # kernel_testが "all dot kernels agree"、tiny modelテストがpassすること
./inkling -m inkling-q3.safetensors -c 2048 --batch 32 --topk 6 -v -p "..."      # 推奨:Q3 + top-6
```

* `-b q3`は3.5bit形式(Q3_0:32重みでfp16スケールを共有)。`-a q4 -e q4`でアテンションと語彙テーブルも縮めます(`-a q8 -e q8`のほうが高品質ですが約0.8GB大きい)。ファイルはmmapされ、常駐が必要なのはアテンション・出力テーブル・キャッシュの約1.5GBだけ。244億のexpert重みはルーターが選んだ分だけSSDから読み込みます。expertがメモリ(プロセスのRAMかcgroup上限)に収まらないときはLRUの**expertキャッシュ**が自動で有効になり(`--cache MB`で指定、`--cache 0`でmmapのみ)、expertを丸ごと大きな並列`pread()`で読みます。ページフォルトで読むより約3倍速く(フォルト1回は4〜16KBしか読まず、ドライブのランダムIOPSで詰まる)、読み込みを減らすのが速度の主なつまみで、推奨は`--topk 6`(expert数を減らす)です。**`--passes 2`/`1`は避けてください**:標準との出力の差が大きく(KL 0.61 / 1.07、top-1一致73% / 61%)なります。
* 実測(24コアx86 Linux、2MBランダム読みで約250MB/sのUSB SSD、プロセスを5GBに制限、ファイルはcold、Q3に自動2.1GBキャッシュ):8 expertで0.57 tokens/s、**`--topk 6`で0.85 tokens/s**(読み込み26%減)。全部ページキャッシュに載れば同じファイルで約3.5 tokens/sです。Q4ファイルとの品質比較(英語・日本語・コード610トークン):Q3はperplexity 3.105(Q4は3.050)、KL 0.060、top-1一致91%。Q3 + `--topk 6`は3.185、KL 0.080、90%。AppleのSSDはこのUSBドライブより数倍速いはずですが、未測定です。プロンプトは`--batch`トークンずつまとめて処理し(expertはチャンクごとに1回だけ読む)、生成より1トークンあたりずっと速くなります。`--prefetch N`(次の層でありそうなexpertN個を裏で読む。層l+1のルーターを層lの入力に当てると、上位8のexpertの77%を当てられる)は、このドライブでは推測の外れ分で読み込みが増えて遅くなったので、既定はオフです。
* Q3_0/Q4_0/Q8_0とfloat内積にNEONカーネル(SDOT)を入れ、パフォーマンスコアだけを使います。aarch64向けにコンパイル確認はしましたが、Apple実機が無い環境で書いたので、実機では`make test`で確認してください(`tests/kernel_test`が全SIMDカーネルをスカラ実装と比較します)。

## 単一ファイルと量子化

`inkling_quantize`はGGMLのQ4_0/Q8_0ブロックを持つsafetensorsを書き出し、トークナイザと`config.json`もU8テンソルとして同じファイルに入れます。

| 部分 | 既定 | 備考 |
|---|---|---|
| ルーティングexpert(244億重み) | Q4_0 (`-b q4`)、`-b q3`で3.5bitのQ3_0 | NVFP4(層0-19)/MXFP8(層20-23)をデコードして再量子化 |
| アテンション行列 | Q8_0 (`-a q8`) | `q4`/`f16`も可 |
| 埋め込み/出力テーブル | Q8_0 (`-e q8`) | パディング除去(200,058行) |
| ノルム、ルーター、相対位置、畳み込み、ループ制御 | F32 | 小さい |

NVFP4もQ4_0も1重みあたり4.5bitなので、expertについては「再量子化」であって、サイズが小さくなるわけではありません。3.5bitのQ3_0が小さいほうの選択肢です。vision towerとMTPヘッドは変換しません。

## 検証

* `make test`:小さなランダムチェックポイント(NVFP4/MXFP8、スライディング/グローバル層、ループ)を作り、変換のデコードとC実装の出力を、独立に書いたPyTorch実装と比較します(26トークン・窓8で約4e-4まで一致)。
* `make test-tokenizer MODEL_DIR=Inkling-2-Mini`:Hugging Face `tokenizers`と25ケース(日本語、絵文字、空白、特殊トークンなど)で完全一致。
* 実モデルの生成品質は、実際に動かして確認する必要があります。
