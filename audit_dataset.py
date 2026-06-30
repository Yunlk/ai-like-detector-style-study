import csv
from collections import Counter

path = "data_pre_llm.csv"

rows = []
with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print("总行数:", len(rows))

years = Counter(row["year"] for row in rows)
print("\n年份分布:")
for year, count in sorted(years.items()):
    print(year, count)

word_counts = []
empty_abstracts = 0
short_abstracts = 0
missing_doi = 0

for row in rows:
    abstract = row.get("abstract", "")
    wc = len(abstract.split())
    word_counts.append(wc)

    if not abstract.strip():
        empty_abstracts += 1
    if wc < 80:
        short_abstracts += 1
    if not row.get("doi", "").strip():
        missing_doi += 1

print("\n摘要长度:")
print("最短:", min(word_counts))
print("最长:", max(word_counts))
print("平均:", round(sum(word_counts) / len(word_counts), 2))

print("\n问题统计:")
print("空摘要:", empty_abstracts)
print("短摘要 <80词:", short_abstracts)
print("缺 DOI:", missing_doi)

print("\n前10条快速检查:")
for row in rows[:10]:
    print("-" * 60)
    print("year:", row["year"])
    print("title:", row["title"][:120])
    print("journal:", row["journal"])
    print("discipline:", row["discipline"])
    print("abstract:", row["abstract"][:250])