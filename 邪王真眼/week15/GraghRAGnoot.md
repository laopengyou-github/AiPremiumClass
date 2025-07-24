mkdir -p ./邪王真眼/week15/ragtest/input
curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt > ./ragtest/input/book.txt
graphrag init --root ./邪王真眼/week15/ragtest
graphrag index --root ./邪王真眼/week15/ragtest

graphrag query --root ./邪王真眼/week15/ragtest --method global --query "文章讲了什么，说中文"
graphrag query --root ./邪王真眼/week15/ragtest --method local --query "军争这部分讲的什么"