#! perl
# weather_makedata.pl weather_sapporo.csv > weather_sapporo.train
# ref. http://arakilab.media.eng.hokudai.ac.jp/~t_ogawa/wiki/index.php?LibSVM

# 気象庁からダウンロードしたCSVファイルを編集して
# 天気をクラスとした学習データを作成する

use strict;
use warnings;

# クラスにしたい列番号 (0~)
my $class_num = 5;

# 天気の列番号（あれば）
my @tenki_id = (5);

########################

my @data;
my %convlist = (
       '晴' => '+1', '曇' => '-1', '雨' => '-1', '雪' => '-1'
);	# 天気からクラスに変換するハッシュ

open(FILE, "<", shift) or die("file open error");
while (<FILE>)
{
	next if ($. <= 5);		# 5行目までは捨てる
	push(@data, [split(/,/)]);	# コンマで分割した無名配列を追加
}
close(FILE);

# 天気の整形 (データ数の関係から，雲後晴 などを 雲 にし，数値に変換)
foreach my $id (@tenki_id)
{
	foreach (@data) {
		my @temp = keys %convlist;	# 正規表現内に展開する用
		$" = '|';			# その際の区切り文字はorにしたい"
		$$_[$id] =~ /(@temp)/;	# /晴|曇|雨|雪/
		$$_[$id] = $convlist {"$1"};
	}
}

# 出力
foreach (@data)
{
	print "$$_[$class_num]";		# はじめにクラスを出力
	for my $i (1 .. @$_ - 1) {		# ここでは日付は用いないので 1個目からloopスタート
		next if ($i == $class_num);	# クラスはもう出力したのでskip
		print " $i:$$_[$i]";		# 素性番号と値を出力
	}
	print "\n";
}
