library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm
library(gtools) # for discretisation
library(corrplot)
library(Hmisc)
library(devtools)
library(PerformanceAnalytics)
library(FactoMineR)
library(fpc)


#Agents Data

Agents <- read.csv("C:/DataAnalysisOperantAI/agents.csv")
head(Agents)
summary(Agents)
####Hypothesis 1: A more experienced agent has a more call frequency###################
#Add a new column of today's date
Agents$TodayDate <- rep(as.Date(Sys.Date()), nrow(Agents))

#Calculate Agents experience

Agents$Experience <-floor(as.numeric(as.Date(Agents$TodayDate)-as.Date(Agents$hire_date))/365.25)
head(Agents)
count <- Agents$Experience
names(count) <- 1001:1055
barplot(count, xlab = "Agents", ylab = "Years of Experience", col = "darkblue")

#Agents Per supervisor

count1 <- table(Agents$id, Agents$supervisor_id)
barplot(count1, col = colors()[c(1,2,3,4,5,6,7,8,9,10)], xlab = "Supervisor ID", ylab = "No of Agents")
count2 <- table(Agents$Experience, Agents$supervisor_id)
barplot(count2, col = colors()[c(1,2,3,4,5,6,7,8,9,10)], beside = T, xlab = "Supervisor ID", ylab = "Years of Experience of Agents")

#Agents Per call center
count3 <- table(Agents$id, Agents$call_centre_id)
barplot(count3, col = colors()[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)], xlab = "Call Center id", ylab = "Agents Per Call Center" )



#Calls Data

calls <- read.csv("C:/DataAnalysisOperantAI/calls.csv")
head(calls)
count4 <- table( calls$debtor_id,calls$agent_id)
barplot(count4, xlab = "Agent ID", ylab = "No of Clients")

summary(calls)

#Merge calls and agents data

agents_calls_data <- merge(Agents, calls, by.x="id", by.y= "agent_id")



######Calls Analysis############

#Calls made by each agent

agentCalls <- as.data.frame(table(calls$agent_id))
agentCalls$agent_id <- 1000:1054
head(agentCalls)
agentCalls <- subset(agentCalls, select = c(Freq, agent_id))
head(agentCalls)
agentCalls <- agentCalls[,c("agent_id", "Freq")]
head(agentCalls)
colnames(agentCalls) <- c("agent_id", "CallsFrequency")
head(agentCalls)
countCalls <- c(agentCalls$Agents, agentCalls$CallsFrequency)
barplot(countCalls, col = 'Red', xlab = "Agents", ylab = "CallFrequency")



#cluster analysis

clus <- kmeans(agentCalls, centers = 3)
plot(agentCalls, col=clus$cluster, lwd = 5)
points(clus$centers, col = 1:2, pch = 8, cex = 2, lwd=5)
plotcluster(agentCalls, clus$cluster)
clusplot(agentCalls, clus$cluster, color = TRUE, shade = TRUE, labels=2, lines=0, lwd=3)




