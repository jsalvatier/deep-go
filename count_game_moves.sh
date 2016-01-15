find $1  -mindepth 2 -maxdepth 2 -type d -exec sh -c 'echo "{}\t$(find "{}" -type f | wc -l)"' \;
