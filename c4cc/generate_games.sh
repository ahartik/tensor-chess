
while true
do

R=`date +%Y%m%d_%H%M%S`
echo $R
TMP=/mnt/tensor-data/c4cc/tmp/games.$R.recordio
../bazel-bin/c4cc/generate_ai_games $TMP 10000
mv $TMP /mnt/tensor-data/c4cc/games/games.$R.recordio

done
