#!/bin/bash

# 人物データファイル定義
PERSON="./person.txt"

# 中間ディレクトリ定義
tmp="../tmp"

# アウトプットディレクトリ定義
out='../data/img_origin/'

# アウトプットディレクトリを絶対パスで取得
SCRIPT_DIR=`dirname $0`
pushd $out
aout=`pwd`
popd

# URLエンコード関数定義
urlencode () {
    echo "$1" | nkf -WwMQ | tr = %
}

# 画像収集関数定義
imageGather () {
    if [ $# -ne 3 ]; then
        return false
    fi

    class=$1
    enName=$2
    jaName=$3

    # 人物名をエンコード
    encodedName=`urlencode $3`
    echo $encodedName

    # URLを作成
    url="https://www.bing.com/images/search?&q="
    url=$url$encodedName

    # テンポラリディレクトリ作成・移動
    mkdir -p $tmp/${class}/${enName}
    pushd $tmp/${class}/${enName}

    # wgetでJPEG画像のみ収集
    wget -r -l 1 -A jpg,JPG,jpeg,JPEG -H \
    -erobots=off \
    --exclude-domains=bing.com,bing.net \
    $url

    find . -type f \( -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.JPEG" \) | \
    awk \
    -v "out=$aout" \
    -v "class=$class" \
    -v "enName=$enName" \
    '{
        command = sprintf("cp %s %s/%s_%s_%05d.jpg", $0, out, class, enName, NR)
        # コマンドを実行して結果を取得
        buf = system(command);
        # stream をclose
        close(command);
    }'

popd

}

# 画像収集実行
while read line; do
    imageGather $line
done  1:
