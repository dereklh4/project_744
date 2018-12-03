while true;
do
	date +%s%N >> $1
	head -n 2 /proc/meminfo >> $1
done
