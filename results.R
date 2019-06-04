# Load data and packages

data <- read.csv('/Users/anhvu/Documents/Master/Thesis Spring 2019/Observations - FINAL_1st.csv',
  stringsAsFactors = FALSE, header = TRUE, skip = 0)
data_2 <- read.csv('/Users/anhvu/Documents/Master/Thesis Spring 2019/Observations - FINAL_2nd.csv',
  stringsAsFactors = FALSE, header = TRUE, skip = 0)
words <- read.csv('/Users/anhvu/Documents/Master/Thesis Spring 2019/words.csv',
  stringsAsFactors = FALSE, header=TRUE, skip = 0)
verbs <- read.csv('/Users/anhvu/Documents/Master/Thesis Spring 2019/verbs.csv',
  stringsAsFactors = FALSE, header=TRUE, skip = 0)
np <- read.csv('/Users/anhvu/Documents/Master/Thesis Spring 2019/ps_np.csv',
  stringsAsFactors = FALSE, header=TRUE, skip = 0)

install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer")

library(ggplot2)
library(dplyr)
library(wordcloud)
library(RColorBrewer)

# Plots on all models on each feature group
plt1 <- data %>%
  mutate(average_f1 = (F1_count + F1_tf)/2) %>%
  filter(group %in% c('Verbs (n=100)')) %>%
  ggplot(aes(x = model, y = F1_count)) + geom_bar(stat = 'identity', 
  fill = 'purple') + xlab('Classifiers') + ylab('F1 score on verbs') + 
  geom_text(aes(label=F1_count), vjust=-0.3, color="Black", size=3.5) 

plt1

# Plots on feature importance
adjs <- c('Adjectives (n = 100)', 'Positive adjectives (n=100)', 
    'Negative adjectives (n=100)')
verbs <- c('Positive verbs (n=100)', 'Negative verbs (n=100)', 'Verbs (n=100)')

grams <- c('Bi-grams (n = 100)','Tri-grams (n = 100)','Unigrams (n = 100)')

phrases <- c('Positive noun phrase ','Negative noun phrase')

plt2 <- data %>%
  filter(group %in% phrases) %>% 
  filter (model != 'Base') %>%
  ggplot(aes(x = group, y = Average.F1, fill = group)) + 
  geom_bar(stat = 'identity') + xlab('Models') + ylab('F1 score') + 
  facet_grid(.~model) + 
  scale_color_manual(values = c("Positive noun phrase" = "orange", 
  "Negative noun phrase" = "red"))  + scale_x_discrete(labels = NULL) 
plt2

set.seed(1234)

wordcloud(words = np$term, freq = np$score, min.freq = 2, 
  random.order = FALSE, max.words = 100, colors=brewer.pal(8, "Dark2"))

wordcloud(words = verbs$term, freq = verbs$score, min.freq = 1, 
          random.order=FALSE, max.words = 100, colors=brewer.pal(8, "Dark2"))

plt3 <- data_2 %>%
  select(n, Feature.set, f1_mean) %>%
  filter(Feature.set %in% c('Positive verbs ', 'Negative verbs ', 'Verbs ')) %>%
  ggplot(aes(x = n, y = f1_mean, color = Feature.set)) + geom_point() + 
  geom_line() + xlab('Number of features') + ylab('F1 score') + 
  scale_x_continuous(breaks=seq(0,1000,200)) +
  scale_color_discrete(name = "Feature group")
plt3
  
plt4 <- data_2 %>%
  select(n, Feature.set, f1_mean) %>%
  filter(Feature.set == ' ') %>%
  ggplot(aes(x = n, y = f1_mean, color = Feature.set)) + geom_point() + 
  geom_line() + xlab('Number of features') + ylab('F1 score') + 
  scale_x_continuous(breaks=seq(0,1000,200)) +
  scale_color_discrete(name = 'Feature group')
plt4

plt5 <- data_2 %>%
  select(n, Feature.set, f1_mean) %>%
  filter(n == 1000) %>%
  ggplot(aes(x = Feature.set, y = f1_mean)) + geom_bar(stat = 'identity') +
  xlab('n=100') + ylab('F1 score') + 
  scale_color_manual(values = c("Positive noun phrase" = "orange", 
  "Negative noun phrase" = "red"))  + scale_x_discrete(labels = NULL) 
plt5