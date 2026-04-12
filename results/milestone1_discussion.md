# Retrieval System Evaluation

## Overview

This document evaluates and compares BM25 keyword-based retrieval and semantic (embedding-based) retrieval on a set of 10 queries across three difficulty levels using the Amazon Books dataset.

- **Easy [E]**: Short keyword queries where exact term matching is likely sufficient
- **Medium [M]**: Queries that require understanding of meaning beyond exact keywords
- **Complex [C]**: Multi-faceted queries with multiple constraints

Both methods retrieved the top-5 results per query.

## Results

Top-5 retrieval results for all 10 queries. Semantic scores are L2 distances (lower = more similar); BM25 scores are relevance scores (higher = more relevant).

---

### Query 1 [E]: "mystery novel"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | Trust Me | 9.19 | The Da Vinci Code | 0.51 |
| 2 | Disappearance of Adèle Bedeau | 8.95 | The Missing American | 0.67 |
| 3 | Seasons | 8.94 | In Peppermint Peril | 0.68 |
| 4 | The Emperor of Ocean Park | 8.70 | Smokescreen (Eve Duncan, 25) | 0.68 |
| 5 | The Fire Thief (A Dark Paradise Mystery) | 8.69 | Crimes of Memory | 0.68 |

---

### Query 2 [E]: "cookbook recipes"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | The Ultimate Paleo Cookbook | 12.80 | 101 One-Dish Dinners | 0.38 |
| 2 | The Adventures of Fat Rice | 12.22 | FlavCity's 5 Ingredient Meals | 0.41 |
| 3 | The McDougall Quick and Easy Cookbook | 11.95 | Plated: Weeknight Dinners, Weekend Feasts | 0.41 |
| 4 | Cooking Know-How | 11.95 | Taste of Home Christmas 2014 | 0.44 |
| 5 | The Viennese Kitchen | 11.72 | Sunday Suppers | 0.45 |

---

### Query 3 [E]: "science fiction space"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | Ender's Game (The Ender Quintet) | 19.75 | Down and Out in the Magic Kingdom | 0.82 |
| 2 | I Am Crying All Inside (Clifford D. Simak) | 14.41 | Ender's Game (The Ender Quintet) | 0.97 |
| 3 | Stars Are Legion | 13.15 | Braking Day | 0.99 |
| 4 | The Garden of Rama | 13.06 | Protector of the Realm | 1.07 |
| 5 | Protector of the Realm | 12.81 | For Love of Mother-Not | 1.09 |

---

### Query 4 [E]: "python programming"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | The D Programming Language | 10.97 | Programming Arduino | 1.08 |
| 2 | Programming the World Wide Web | 10.88 | Genetic Algorithms and ML for Programmers | 1.12 |
| 3 | Definitive XML Application Development | 10.82 | Sqr in PeopleSoft and Other Applications | 1.21 |
| 4 | Programming Arduino | 10.02 | DKfindout! Coding | 1.25 |
| 5 | The Official BBC micro:bit User Guide | 9.70 | Beginner's Step-by-Step Coding Course | 1.25 |

---

### Query 5 [M]: "book to help with anxiety"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | Anxiety Relief for Teens | 23.97 | Everyone in This Room Will Someday Be Dead | 0.58 |
| 2 | Everyone in This Room Will Someday Be Dead | 22.75 | Anxiety Relief for Teens | 0.62 |
| 3 | Zendoodle Colorscapes: Outrageous Owls | 20.31 | God Gave Us You | 0.84 |
| 4 | Get the Guy | 18.96 | The Supreme Word Search Book for Adults | 0.89 |
| 5 | The Hoarder in You | 18.48 | Paying with Their Bodies | 0.90 |

---

### Query 6 [M]: "story about finding yourself"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | Eight Hundred Grapes: A Novel | 11.79 | The Way to Bea | 0.90 |
| 2 | The Whispering Soul | 11.48 | Don't Look for Me: A Novel | 0.96 |
| 3 | An Embarrassment of Witches | 11.14 | It Had To Be You | 1.00 |
| 4 | Spaghetti in a Hot Dog Bun | 10.57 | My Name Is Nathan Lucius | 1.01 |
| 5 | Distant Shores, Silent Thunder | 10.54 | Charleston: A Novel | 1.04 |

---

### Query 7 [M]: "guide for first time parents"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | Frommer's Alaska | 13.57 | Mac and Cheese (I Can Read Level 1) | 0.91 |
| 2 | The Jumbo Book of Art | 13.52 | Sleeping Bags To S'mores: Camping Basics | 0.91 |
| 3 | Lonely Planet Cruise Ports Alaska | 13.25 | The Best Seat in Second Grade | 0.92 |
| 4 | DK Eyewitness Italy: 2020 | 12.86 | Lessons Learned: The Kindergarten Survival Guide | 0.93 |
| 5 | Latin Primer 1 (Teacher Edition) | 12.70 | Regret Free Parenting | 0.94 |

---

### Query 8 [C]: "best book to learn machine learning with no math background"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | Genetic Algorithms and ML for Programmers | 28.12 | Genetic Algorithms and ML for Programmers | 0.92 |
| 2 | My Kindergarten Math Workbook | 25.42 | Pre-Algebra Essentials For Dummies | 0.94 |
| 3 | Pre-Algebra Essentials For Dummies | 24.23 | Bead Jewelry Making for Beginners | 0.97 |
| 4 | Piece in the Hoop | 23.92 | From the Moon, I Come in Peace | 1.01 |
| 5 | The Sewing Machine Accessory Bible | 23.74 | 101 Careers in Mathematics | 1.02 |

---

### Query 9 [C]: "historical fiction set in world war 2 from a female perspective"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | The Forgotten Room | 33.87 | The Women Who Flew for Hitler | 0.57 |
| 2 | Daughter of a Daughter of a Queen | 29.00 | Lilac Girls: A Novel | 0.59 |
| 3 | The Wind Is Not a River | 27.00 | The Alice Network | 0.64 |
| 4 | Age 14 | 26.32 | Dear Mrs. Bird: A Novel | 0.66 |
| 5 | The Long-Lost Secret Diary of the World's Worst Dinosaur Hunter | 25.93 | Dragon Harvest (The Lanny Budd Novels) | 0.69 |

---

### Query 10 [C]: "self help book for overcoming procrastination and building better habits"

| Rank | BM25 Title | BM25 Score | Semantic Title | Semantic Score |
|------|-----------|------------|----------------|----------------|
| 1 | Mini Habits for Teens | 24.77 | The 5-Minute Productivity Journal | 0.78 |
| 2 | The Fix: How Nations Survive and Thrive | 22.85 | Live With Purpose: Creating Positive, Lasting Change | 0.83 |
| 3 | Live With Purpose: Creating Positive, Lasting Change | 22.83 | The Man Who Wanted to Be Happy | 0.86 |
| 4 | The Man Who Wanted to Be Happy | 22.43 | If You Can Talk You Can Write | 0.94 |
| 5 | The Dad Connection | 22.24 | Five Acres and Independence | 0.95 |

## Method Comparison

We picked five queries from our set to look into more closely and compare how each method performed.

---

### Discussion: Query 4 [E] : "python programming"

**BM25** returned "The D Programming Language" at the top. It picked up on the word "programming" but had no Python-specific books to show.

**Semantic search** also came up empty for Python specifically, returning Arduino and machine learning titles instead. It understood the general topic of programming but couldn't find Python books that weren't there.

**Verdict**: Both methods failed here, which is surprising. Perhaps our 20,000 stratified samples don't contain a single book about Python! If there really are no Python books then the results we have here are understandable.

---

### Discussion: Query 5 [M] : "book to help with anxiety"

**BM25** did well finding "Anxiety Relief for Teens" due to the word "anxiety" appearing directly. "Everyone in This Room Will Someday Be Dead" may be a reasonable find. At this stage, we realise it may be useful to have a summary or abstract of the book to show (so we can better assess the result).

**Semantic search** returned the same two top books but in reverse order, putting the novel first. It returned a word search book, which is likely due to a user with anxiety issues finding the book useful.

**Verdict**: BM25 is slightly ahead here. Having the word "anxiety" in the query was enough to directly put a book about anxiety at the top.

---

### Discussion: Query 7 [M] : "guide for first time parents"

**BM25** completely missed the point. It saw the word "guide" and returned mostly travel guides. It had no idea the query was about parenting.

**Semantic search** did much better. "Lessons Learned: The Kindergarten Survival Guide for Parents" and "Regret Free Parenting" seem like relevant books. At least it understood the parenting theme, even though the word "parenting" never appeared in the query.

**Verdict**: A clear win for semantic search. BM25 got completely thrown off by the word "guide", while semantic search at least understood what the user was actually asking for.

---

### Discussion: Query 10 [C]: "self help book for overcoming procrastination and building better habits"

**BM25** picked up on words like "habits" and returned some self-help adjacent books, but also pulled in completely unrelated titles like "The Dad Connection", likely because some common words in the query showed up in unrelated reviews.

**Semantic search** did slightly better, leading with "The 5-Minute Productivity Journal". The other books look like self help books but they don't seem to address both procrastination and building better habits together.

**Verdict**: This complex query shows that when someone is looking for something very specific, both methods can only get you partway there.

---

### Discussion: Query 8 [C] : "best book to learn machine learning with no math background"

**BM25** started well with "Genetic Algorithms and ML for Programmers" at the top, but then it also returned "My Kindergarten Math Workbook" (matched "math") and "The Sewing Machine Accessory Bible" (both matched "machine" as in sewing machine). The word "machine" in "machine learning" confused it badly.

**Semantic search** also found "Genetic Algorithms and ML for Programmers" first and "Pre-Algebra Essentials For Dummies" second. The rest of its results weren't great either, but at least it didn't return sewing books. It understood that "machine learning" is one concept.

**Verdict**: Semantic search did better here. BM25's habit of treating every word independently really hurt it.

---

## Findings

**BM25 strengths:**

- Great for short, simple searches: if you type exactly what's in the book title or review, BM25 will find it quickly (Queries 1, 2, 3)
- Very fast and easy to understand why a result was returned

**BM25 weaknesses:**

- Gets confused by words with multiple meanings. For example, searching "guide" returns travel guides, and "machine" returns sewing machine books (Queries 7, 8)
- Treats every word in a long query independently, so it can miss the overall point of what you're looking for
- Common words like "book", "help", or "best" often match irrelevant results and drag down quality

**Semantic search strengths:**

- Understands what you mean, not just what you typed. For example, it found parenting books for "guide for first time parents" even without the word "parenting" (Query 7)
- Not confused by words that sound similar but mean different things (e.g. "machine learning" vs sewing machines)

**Semantic search weaknesses:**

- Can be less precise than BM25 for short, direct keyword searches (Query 5)

**Where both methods struggle:**

- When the book simply doesn't exist in the dataset (Query 4: "python programming")
- Vague or abstract queries (Query 6: "story about finding yourself"), and neither method handles these well
- Long queries with many specific requirements, and both methods can only partially match what the user is looking for

## Recommendations

- **Combine both methods**: Using BM25 and semantic search together would give better results than either alone. BM25 nails exact keyword matches while semantic search covers the gaps where meaning matters more than exact wording.
- **Add more data**: As seen with Query 4, no retrieval method can find a book that isn't there. Expanding the dataset to cover more topics would have a bigger impact than any algorithm change.

## Conclusion

BM25 and semantic search each shine in different situations. BM25 is fast and reliable when users know exactly what they're looking for, but falls apart when words have multiple meanings or when the query is complex. Semantic search is better at understanding intent and handling nuanced queries, but is no better than BM25 when the relevant books are simply not in the dataset. The clearest takeaway is that combining both approaches would give the best results across the widest range of queries.
