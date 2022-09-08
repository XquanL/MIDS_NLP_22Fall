ad <- read.csv("Advertising.csv")
head(ad)
str(ad)
summary(ad$sales)
hist(ad$sales)
boxplot(ad$sales)


summary(ad$newspaper)
par(mfrow = c(1,2))
hist(ad$newspaper, main = 'Histogram of Newspaper sale', xlab = 'Newspaper')
boxplot(ad$newspaper)

#SLR model

slrmod <- lm(sales ~ newspaper, data = ad)
summary(slrmod)  #t^2 = f 在simple linear regression中 and there is only one variable

#BOSTON DATA
install.packages('ISLR2')
library(ISLR2)
boston <- Boston
summary(boston)
head(boston)

tinytex::install_tinytex()
