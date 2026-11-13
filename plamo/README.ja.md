# PLaMo 3 Translate (Cat's eye)

Preferred Networks の翻訳特化LLM [PLaMo 3 Translate](https://huggingface.co/collections/pfnet/plamo-3-translate) を、C(依存ライブラリなし)で動かす翻訳CLIです。標準では最小の **0.8B**(53言語)を使い、量子化でさらに小さくできます。

PLaMo固有の部分(設定、重み名、サンドイッチnormのデコーダ層、PFML3プロンプト)だけが`plamo/`にあり、汎用部品は`../catseye.h`に入っています:
safetensors読み込み(F32/F16/BF16/Q4_0/Q8_0)、量子化GEMM、YaRN RoPE、KVキャッシュ付きGQAアテンション、オフセット付きRMSNorm、Unigramトークナイザ。

## 使い方

```sh
make                                     # plamo と plamo_quantize
hf download pfnet/plamo-3-translate-0.8b --local-dir hf-0.8b
./plamo_quantize -i hf-0.8b -o models/q4          # 482 MB (元は1.78 GB)

./plamo -m models/q4 -t en -p "今日はとても良い天気ですね。"
echo "Hello, world." | ./plamo -m models/q4 -t ja --polite
./plamo -m models/q4 -t ja -g "catseye=キャッツアイ" -p "I use catseye every day."
cat doc.txt | ./plamo -m models/q4 -t en -l       # 1行ずつ訳す
```

オプションは`./plamo`で表示します。`-s`で原文の言語、`-c`で参照文脈、`-g 原語=訳語`で用語集、`--polite`/`--plain`で日本語の文体、`-v`で速度表示です。言語コードは`en` `ja` `zh-hans` `zh-hant` `ko` `fr` `de` `ja-easy`などの小文字です。

## サイズを小さくする

`plamo_quantize -b 本体 -e 埋め込み -h 出力層`(型は`f16` `q8` `q4`)。0.8Bでの大きさ:

| 構成 | サイズ | 備考 |
|---|---|---|
| 元のBF16 | 1.78 GB | 公式の重みをそのまま読めます |
| `-b q8` | 906 MB | BF16とほぼ同じ訳 |
| `-b q4 -e q8 -h q8` | 639 MB | 4bit本体、語彙表は8bit |
| `-b q4`(標準) | 482 MB | 訳語がときどき別の言い方になる程度 |

これ以上小さくするなら、語彙表(107,520語×1536次元の埋め込みと出力層で全体の4割弱)を使う言語だけに絞る方法が次の候補です(未実装)。`--fast`はGGML標準のQ4_0(量子化は速いが少し劣ります)。

## 検証

`tests/check_reference.py`は公式のHugging Face実装と、トークナイザのID一致と貪欲デコードの訳文を比べます(torch、transformers、numbaが必要)。

```sh
cd tests && python3 check_reference.py ../hf-0.8b ../models/q4
```

BF16版ではトークナイザは全ケースで完全一致し、訳文は6件中5件が一致しました(残りの1件は公式側のbf16演算の丸めによる僅差の違いです)。共通部品のテストは`make test-core`(`../test/test_catseye_core.c`)です。

## ライセンス

重みは[PLaMoコミュニティライセンス](https://plamo.preferredai.jp/)です。個人・学習・非商用(学術含む)は自由、年間売上10億円以下の商用利用は登録が必要、それを超える場合はPFNとの契約が必要です。モデルカードで最新の条件を確認してください。
