rm(list=ls())
setwd("D:/DS/Datasets/EmployeeAbsentism_Edwisor")
getwd()


install.packages(c("dplyr","DMwR","ggplot2"))
library("dplyr") #used for manipulationg select arrange
library("DMwR") #used for msining with R
library("ggplot2") #used for plotting the graphs
install.packages("DMwR")
library("VIM")

#reading file df
df = read.csv("Absent.csv", header = TRUE)

#replacing missing values with data
for(i in 1:ncol(df)){
  df[is.na(df[,i]), i] <- mean(df[,i], na.rm = TRUE)
}

#Dropping highly correlated values

df$Weight<- NULL
df$Age<- NULL

#Transforming data to time series data

df1 = df[df$Month.of.absence == 1, ]
df2 = df[df$Month.of.absence == 2, ]
df3 = df[df$Month.of.absence == 3, ]
df4 = df[df$Month.of.absence == 4, ]
df5 = df[df$Month.of.absence == 5, ]
df6 = df[df$Month.of.absence == 6, ]
df7 = df[df$Month.of.absence == 7, ]
df8 = df[df$Month.of.absence == 8, ]
df9 = df[df$Month.of.absence == 9, ]
df10 = df[df$Month.of.absence == 10, ]
df11 = df[df$Month.of.absence == 11, ]
df12 = df[df$Month.of.absence == 12, ]

#Create new dataframe for time series analysis

new = data.frame()

for (i in 0:49){
  new <- rbind(new, df1[i, ])
  new <- rbind(new, df2[i, ])
  new <- rbind(new, df3[i, ])
  new <- rbind(new, df4[i, ])
  new <- rbind(new, df5[i, ])
  new <- rbind(new, df6[i, ])
  new <- rbind(new, df7[i, ])
  new <- rbind(new, df8[i, ])
  new <- rbind(new, df9[i, ])
  new <- rbind(new, df10[i, ])
  new <- rbind(new, df11[i, ])
  new <- rbind(new, df12[i, ])
  
}

#Save id and drop it

id = data.frame()
id = new$ID
new$ID<- NULL

#Plot to check moving average
install.packages("forecast")
library(forecast)
trend = ma(new$Absenteeism.time.in.hours, order = 12,centre = T)
plot(as.ts(trend))
lines(trend)
#Decompose seasonality
#install.packages("fpp")
library(fpp)
ts_data = ts(new$Absenteeism.time.in.hours, frequency = 12, start = 1)
decompose_data = decompose(ts_data, "additive")
adjust_data = ts_data-decompose_data$seasonal
plot(adjust_data)


##Creating the ARIMA model
#Generating values for ARIMA
#Getting the auto correlation factor
p = diff(log(adjust_data)) #Differentiating 
l = log(adjust_data)
is.na(p) <- sapply(p, is.infinite)          #Removing Inf values
is.na(l) <- sapply(l, is.infinite)

sum(is.na(l))                  

p<- na.omit(p)         #Removing NA values
l<- na.omit(l)
acf(p)     #Autocorrelation factor = 0

pacf(p)    #pacf factor = 5

length(l)


fit<- arima(l, c(5,1,0), seasonal = list(order = c(5,1,0), period = 12))
pred<- predict(fit, n.ahead = 49*12)


#Plot to check moving average
#install.packages("forecast")
library(forecast)
trend = ma(pred, order = 12,centre = T)
plot(as.ts(trend))
lines(trend)
#Decompose seasonality
#install.packages("fpp")
library(fpp)
ts_data = ts(pred, frequency = 12, start = 1)
decompose_data = decompose(ts_data, "additive")
adjust_data = ts_data-decompose_data$seasonal
plot(adjust_data)

#Creating final dataframe for storing predicted values

final = data.frame()
final = as.data.frame(adjust_data)
final$id <- paste(final$x, id)

final$pred<- NULL

final$x<- as.integer(final$x)
final$month <- paste(new$Month.of.absence)

final$x[final$x < 0] <- 0 # Converting any -ve values to 0

#Calculating errors : Mean Average Percent Error and r squared
mape <- (sum(abs(final$x-new$Absenteeism.time.in.hours))/sum(abs(new$Absenteeism.time.in.hours)))*100  #23.28% error
rsq <- cor(final$x, new$Absenteeism.time.in.hours)^2 #0.977 = 97.7% accuracy

#Plotting
library(ggplot2)
theme_set(theme_minimal())
ggplot() + 
  geom_point(data = final, aes(x = month, y = x), color = "red") +
  geom_point(data = new, aes(x = Month.of.absence, y = Absenteeism.time.in.hours), color = "blue") +
  ylab('Absenteeism.time.in.hours') +
  xlab('Month')+
  geom_path()

ggplot() + 
  geom_count(data = final, aes(x = month, y = x), color = "red") +
  geom_count(data = new, aes(x = Month.of.absence, y = Absenteeism.time.in.hours), color = "blue") +
  ylab('Absenteeism.time.in.hours') +
  xlab('Month')+
  geom_path()



write.csv(final, "predictions_timeseries.csv", row.names = F)

