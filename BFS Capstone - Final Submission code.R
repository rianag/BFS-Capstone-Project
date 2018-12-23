#===============================================================================================
#***********************************************************************************************
#*********************************** BFS Capstone Project **************************************
#***********************************************************************************************
#===============================================================================================

#Team members:
  #1. Piyush Gaur
  #2. Priya Gupta
  #3. Ria Nag
  #4. Sahana K

#-------------------------------------------------------------------------------#
# Objective:

# To help CredX, a leading credit card provider, to identify the right customers using predictive models
# using the past data of the bank's applicants and to determine the factors affecting credit risk, 
# create strategies to mitigate the acquisition risk and assess the financial benefit of the project.   

#-------------------------------------------------------------------------------#

# Setting up work directory

setwd("C:/Data_Science/# A BFS Capstone Project/Inputs")

rm(list=ls())

#-------------------------------------------------------------------------------#
#Install and Load the required packages
#-------------------------------------------------------------------------------#

install.packages("Information")
install.packages("gridExtra")
install.packages("grid")
install.packages("ggplot2")
install.packages('corrplot')
install.packages("dplyr")
install.packages("caret")
install.packages("car")
install.packages("MASS")
install.packages("ROSE")
install.packages("woeBinning")
install.packages("e1071")
install.packages("cowplot")
install.packages('caTools')
install.packages("foreach")
install.packages('randomForest')
install.packages("lift")
install.packages('plyr')
install.packages('kernlab')
install.packages('readr')
install.packages('xgboost')

library(Information)
library(gridExtra)
library(grid)
library(ggplot2)
library(corrplot)
library(dplyr)
library(caret)
library(car)
library(MASS)
library(woeBinning)
library(e1071)
library(cowplot)
library(caTools)
library(ROCR)
library(AUC)
library(ROSE)
library(randomForest)
library(lift)
library(plyr)
library(kernlab)
library(readr)
library(foreach)
library(xgboost)

#################################################################################
#Data loading, initial validation and preparation
#################################################################################

# Importing the two CSV files
CreditBureau <- read.csv("Credit Bureau data.csv",na.strings = c("NA",""))
demographics <- read.csv("Demographic data.csv",  na.strings = c("NA",""))

dim(CreditBureau)     #71295 obs. of  19 variables
dim(demographics)     #71295 obs. of  12 variables

str(CreditBureau)     #There are 18 independent variable and 1 dependent variable "Performance.Tag", 
                      #and all variables are of integer type

str(demographics)     #There are 11 independent variabe and 1 dependent variable "Performance.Tag", 
                      #with 7 integer and 5 categorical variables.


# Checking if the key field - Application ID is unique
sapply(list(CreditBureau,demographics),function(x) sum(!duplicated(x$Application.ID)))
  # [1] 71292 71292 (Has 3 duplicate application IDs)

CreditBureau[duplicated(CreditBureau$Application.ID),]
demographics[duplicated(demographics$Application.ID),]
  #765011468                                           
  #653287861                                            
  #671989187  

#Removing both entries of duplicate application ids as it is not possible to identify the correct entry among duplicates
CreditBureau <- CreditBureau[!CreditBureau$Application.ID %in% c(765011468, 653287861, 671989187),]
demographics <- demographics[!demographics$Application.ID %in% c(765011468, 653287861, 671989187),]

#-------------------------------------------------------------------------------#
# Identifying Data quality issues
#-------------------------------------------------------------------------------#

# Identify which columns have how many NA's
sapply(list(demographics,CreditBureau), function(x) length(which(is.na(x))))
  #[1] 1577 3028

colSums(is.na(demographics))
  # Gender - 2
  # Marital.Status..at.the.time.of.application - 6
  # No.of.dependents - 3
  # Education -  119
  # Profession - 14
  # Type.of.residence - 8
  # Performance.Tag - 1425

colSums(is.na(CreditBureau))
  # Avgas CC Utilization in last 12 months - 1058
  # No of trades opened in last 6 months - 1
  # Presence.of.open.home.loans - 272
  # Outstanding.Balance - 272
  # Performance.Tag - 1425

nrow(CreditBureau[which(is.na(CreditBureau$Performance.Tag)),])/nrow(CreditBureau) 
  # There are 1425 records with no performance tag in both datasets which indicates that the applicant is not given credit card.
  # Only 1.99 % of the obs has NA values for - 'perfromance.tag'. 
  # Since Performance.Tag is the target variables, rows with NA's are removed and saved separately for future use.

#Saving performance tag rejected records in a separately file.
rejected_credit <- CreditBureau[which(is.na(CreditBureau$Performance.Tag)),]
rejected_demo <- demographics[which(is.na(demographics$Performance.Tag)),]

CreditBureau<-CreditBureau[!(is.na(CreditBureau$Performance.Tag)),]
demographics<-demographics[!(is.na(demographics$Performance.Tag)),]

nrow(CreditBureau)
nrow(demographics)
  # 69864 obs

# Taking backup of the datasets
credit_wo_woe <- CreditBureau
demo_wo_woe <- demographics

#-------------------------------------------------------------------------------#
# WOE Analaysis and IV (Information Value)
#-------------------------------------------------------------------------------#

# This method is used to replace the missing values with woe values and find significant variables.

# ************** Demographic data *************

#Since Information package treats 1 as 'good', we are adding a new variable 'Reverse.Performance.Tag'.
demographics$Reverse.Performance.Tag <- ifelse(demographics$Performance.Tag == 0,1,0)

##### creating woe buckets for numerical variables

#making a dataset containing character variables with missing values along with "Reverse.Performance.Tag"
colnames <- c("Reverse.Performance.Tag","Marital.Status..at.the.time.of.application.","Gender","Education","Profession","Type.of.residence")
char_missing <- demographics[,colnames(demographics)%in%colnames]
binning <- woe.binning(char_missing, 'Reverse.Performance.Tag', char_missing)
df.with.binned.vars.added <- woe.binning.deploy(char_missing, binning,add.woe.or.dum.var='woe')
demo_woe <-df.with.binned.vars.added[,c(8,10,12,14,16)]

#replacing 5 char variables having missing values with their respective WOE values.
colnames <- c("Marital.Status..at.the.time.of.application.","Gender","Education","Profession","Type.of.residence")
demographics <- demographics[,!colnames(demographics)%in%colnames]
demographics <- cbind(demographics,demo_woe)

#____________________________________________

##### Information value (IV) Analysis
IV_demo <- Information::create_infotables(data=demographics, y="Reverse.Performance.Tag",bins = 2, parallel = TRUE)
IV_demo$Summary

#REPLACING the actual values of No.of.dependents with corresponding WOE VALUES
IV_demo$Tables[3]

  #No.of.dependents     N      Percent          WOE           IV
  #1               NA     3 4.293873e-05  0.000000000 0.000000e+00
  #2            [1,3] 45989 6.582650e-01  -0.005418527 1.937496e-05
  #3            [4,5] 23872 3.416921e-01  0.010382663 5.603467e-05

demographics$No.of.dependents[which(is.na(demographics$No.of.dependents))]<-0
for(i in 1:nrow(demographics))
{
  
  if(demographics$No.of.dependents[i]>=1 & demographics$No.of.dependents[i]< 4)
  {
    if(demographics$No.of.dependents[i]<=3)
    {demographics$No.of.dependents[i]<- -0.005}
  }
  else if (demographics$No.of.dependents[i]>= 4)
  {demographics$No.of.dependents[i]<- 0.01}
  else{demographics$No.of.dependents[i]<- 0}
}

summary(demographics$No.of.dependents)   

#  Information value (IV) Analysis without specifying number of bins
#so that iv is calculated on maximum bins possible for monotonically changing WOE values across bins for continous variables
IV_demo <- Information::create_infotables(data=demographics, y="Reverse.Performance.Tag", parallel = TRUE)
IV_demo$Summary

#Based on IV values, below are the top 3 significant variables.

  #Variable                               IV
  #--------                               ----   
  #No.of.months.in.current.residence     7.895394e-02
  #Income                                4.241078e-02
  #No.of.months.in.current.company       2.176071e-02


# ************* Credit Bureau data *************

#Since Information package treats 1 as 'good', we are adding a new variable 'Reverse.Performance.Tag'.
CreditBureau$Reverse.Performance.Tag <- ifelse(CreditBureau$Performance.Tag == 0,1,0)

##### binning "Avgas.CC.Utilization.in.last.12.months"
colnames<- c("Reverse.Performance.Tag","Avgas.CC.Utilization.in.last.12.months")
char_missing <- CreditBureau[,names(CreditBureau)%in%colnames]
binning_Avgas.CC.Utilization.in.last.12.months <- woe.binning(char_missing, 'Reverse.Performance.Tag', "Avgas.CC.Utilization.in.last.12.months",stop.limit=0.01)
binning_Avgas.CC.Utilization.in.last.12.months[[2]]
char_missing <- woe.binning.deploy(char_missing, binning_Avgas.CC.Utilization.in.last.12.months,add.woe.or.dum.var='woe')
CreditBureau$Avgas.CC.Utilization.in.last.12.months<-char_missing$woe.Avgas.CC.Utilization.in.last.12.months.binned

#binning all other variables with missing values in Credit Bureau data
colnames<- c("Presence.of.open.home.loan","Reverse.Performance.Tag","No.of.trades.opened.in.last.6.months","Outstanding.Balance")
char_missing <- CreditBureau[,names(CreditBureau)%in%colnames]
binning_b <- woe.binning(char_missing, 'Reverse.Performance.Tag', char_missing)
char_missing <- woe.binning.deploy(char_missing, binning_b,add.woe.or.dum.var='woe')

#replacing all other missing variables in Credit Bureau data with their WOE values
CreditBureau$No.of.trades.opened.in.last.6.months<-char_missing$woe.No.of.trades.opened.in.last.6.months.binned
CreditBureau$Presence.of.open.home.loan<-char_missing$woe.Presence.of.open.home.loan.binned
CreditBureau$Outstanding.Balance<-char_missing$woe.Outstanding.Balance.binned

colSums(is.na(demographics))   #0
colSums(is.na(CreditBureau))   #0

#_____________________________________________

##### Information value (IV) Analysis
IV_credit <- create_infotables(data=CreditBureau, y="Reverse.Performance.Tag", parallel = TRUE)
IV_credit$Summary

#All variables except the following 6 variales are monotonically changing across bins:
#"No.of.trades.opened.in.last.12.months"
#"No.of.PL.trades.opened.in.last.6.months"
#"No.of.PL.trades.opened.in.last.12.months"
#"No.of.Inquiries.in.last.6.months..excluding.home...auto.loans."
#"No.of.Inquiries.in.last.12.months..excluding.home...auto.loans."
#"Total.No.of.Trades"
# So we will have to make coarse bins for these 6 variables

colnames<-c("No.of.trades.opened.in.last.12.months","No.of.PL.trades.opened.in.last.6.months",
            "No.of.PL.trades.opened.in.last.12.months",
            "No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.",
            "No.of.Inquiries.in.last.12.months..excluding.home...auto.loans.",
            "Total.No.of.Trades")
IV_coarse_bins<-CreditBureau[,colnames(CreditBureau)%in%colnames]
IV_coarse_bins<-cbind(IV_coarse_bins,CreditBureau$Reverse.Performance.Tag)
IV_coarse <- create_infotables(data=IV_coarse_bins, y="CreditBureau$Reverse.Performance.Tag",bins=3, parallel = TRUE)
IV_coarse$Summary

IV_fine_bins<-CreditBureau[,!colnames(CreditBureau)%in%colnames]
IV_fine_bins<-cbind(IV_fine_bins,CreditBureau$Reverse.Performance.Tag)
IV_fine <- create_infotables(data=IV_fine_bins, y="CreditBureau$Reverse.Performance.Tag", parallel = TRUE)
IV_fine$Summary

IV_final<-rbind(IV_coarse$Summary,IV_fine$Summary)
IV_final<-IV_final[-c(19,20),]
IV_final_sorted<-IV_final[order(IV_final$IV,decreasing = T),]
head(IV_final_sorted)

IV_final_sorted<-rbind(IV_final_sorted,IV_demo$Summary)
IV_final_sorted<-IV_final_sorted[order(IV_final_sorted$IV,decreasing = T),]
head(IV_final_sorted)

#Based on IV values, below are the top significant variables.

  # Variable                                                           IV
  #No.of.Inquiries.in.last.12.months..excluding.home...auto.loans. 0.2715447
  #                         Avgas.CC.Utilization.in.last.12.months 0.2607554
  #                   No.of.times.30.DPD.or.worse.in.last.6.months 0.2415627
  #                  No.of.times.90.DPD.or.worse.in.last.12.months 0.2138748
  #                   No.of.times.60.DPD.or.worse.in.last.6.months 0.2058339

#On the combined demographic and credit bureau data the iv values are highest
#storing the IV values of all variables in a csv file
write.csv(IV_final_sorted,"iv.csv")


#################################################################################
# Exploratory data analysis
#################################################################################

################## UNIVARIATE ANALYSIS ####################

# *** Demographic data ***

boxplot(demo_wo_woe$Income)
median(demo_wo_woe$Income)
#Median income of applicants is 27 units and there are no outliers

boxplot(demo_wo_woe$Age)
median(demo_wo_woe$Age)
#Median Age of applicants is 45 years and there are few outliers

boxplot(demo_wo_woe$No.of.months.in.current.residence)
median(demo_wo_woe$No.of.months.in.current.residence)
# Median No.of.months.in.current.residence is 10 and there are no outliers

boxplot(demo_wo_woe$No.of.months.in.current.company)
median(demo_wo_woe$No.of.months.in.current.company)
# Median No.of.months.in.current.company is 34 and there are few outliers

boxplot(demo_wo_woe$No.of.dependents)
median(demo_wo_woe$No.of.dependents,na.rm = T)
#Median mnumber of dependants is 3 and there are no outliers

ggplot(demo_wo_woe,aes(x=Gender))+geom_bar(fill="red")
#There are more male applicants than Female applicants.

ggplot(demo_wo_woe,aes(x=Marital.Status..at.the.time.of.application.))+geom_bar(fill="red")
#There are more married Applicants than Single applicants.

ggplot(demo_wo_woe,aes(x=Education))+geom_bar(fill="red")
#Most Applicants have Professional education or masters education

ggplot(demo_wo_woe,aes(x=Profession))+geom_bar(fill="red")
#Most applicants have profession SAL(salaried)

ggplot(demo_wo_woe,aes(x=Type.of.residence))+geom_bar(fill="red")
#Most applicants have rented Type.of.residence

#___________________________________________

# *** Credit bureau data data ***

boxplot(CreditBureau$No.of.times.90.DPD.or.worse.in.last.6.months)
median(CreditBureau$No.of.times.90.DPD.or.worse.in.last.6.months,na.rm = T)
# Median is zero and there are few outliers for No.of.times.90.DPD.or.worse.in.last.6.months

boxplot(CreditBureau$No.of.times.60.DPD.or.worse.in.last.6.months)
median(CreditBureau$No.of.times.60.DPD.or.worse.in.last.6.months,na.rm = T)
# Median is zero and there are  outliers for No.of.times.60.DPD.or.worse.in.last.6.months

boxplot(CreditBureau$No.of.times.30.DPD.or.worse.in.last.6.months)
median(CreditBureau$No.of.times.30.DPD.or.worse.in.last.6.months,na.rm = T)
# Median is zero and there are  outliers for No.of.times.30.DPD.or.worse.in.last.6.months

boxplot(CreditBureau$No.of.times.90.DPD.or.worse.in.last.12.months)
median(CreditBureau$No.of.times.90.DPD.or.worse.in.last.12.months,na.rm = T)
# Median is zero and there are  outliers for No.of.times.90.DPD.or.worse.in.last.12.months

boxplot(CreditBureau$No.of.times.60.DPD.or.worse.in.last.12.months)
median(CreditBureau$No.of.times.60.DPD.or.worse.in.last.12.months,na.rm = T)
# Median is zero and there are  outliers for No.of.times.60.DPD.or.worse.in.last.12.months

boxplot(CreditBureau$No.of.times.30.DPD.or.worse.in.last.12.months)
median(CreditBureau$No.of.times.30.DPD.or.worse.in.last.12.months,na.rm = T)
# Median is zero and there are  outliers for No.of.times.30.DPD.or.worse.in.last.12.months

boxplot(credit_wo_woe$Avgas.CC.Utilization.in.last.12.months,names=c("Avgas.CC.Utilization.in.last.12.months"))
median(credit_wo_woe$Avgas.CC.Utilization.in.last.12.months,na.rm = T)
#Median number of avg credit card usage is 15 and there are few outliers

boxplot(credit_wo_woe$No.of.trades.opened.in.last.6.months)
median(credit_wo_woe$No.of.trades.opened.in.last.6.months,na.rm = T)
#Median is 2 and there are  outliers for No.of.trades.opened.in.last.6.months

boxplot(CreditBureau$No.of.trades.opened.in.last.12.months)
median(CreditBureau$No.of.trades.opened.in.last.12.months,na.rm = T)
#Median is 4 and there are  outliers for No.of.trades.opened.in.last.12.months

boxplot(CreditBureau$Total.No.of.Trades)
median(CreditBureau$Total.No.of.Trades,na.rm = T)
#Median is 6 and there are outliers for Total.No.of.Trades

boxplot(CreditBureau$No.of.PL.trades.opened.in.last.12.months)
median(CreditBureau$No.of.PL.trades.opened.in.last.12.months,na.rm = T)
#Median is 2 and there are  outliers for No.of.PL.trades.opened.in.last.12.months

boxplot(CreditBureau$No.of.PL.trades.opened.in.last.6.months)
median(CreditBureau$No.of.PL.trades.opened.in.last.6.months,na.rm = T)
#Median is 1 and there are outliers for No.of.PL.trades.opened.in.last.6.months

boxplot(CreditBureau$No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.)
median(CreditBureau$No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.,na.rm = T)
#Median is 1 and there are  outliers for No.of.Inquiries.in.last.6.months..excluding.home...auto.loans

boxplot(CreditBureau$No.of.Inquiries.in.last.12.months..excluding.home...auto.loans.)
median(CreditBureau$No.of.Inquiries.in.last.12.months..excluding.home...auto.loans.,na.rm = T)
#Median is 3 and there are  outliers for No.of.Inquiries.in.last.12.months..excluding.home...auto.loans

boxplot(credit_wo_woe$Outstanding.Balance)
median(credit_wo_woe$Outstanding.Balance,na.rm = T)
# Median is 774242 for Outstanding.Balance

ggplot(credit_wo_woe,aes(x=factor(Presence.of.open.auto.loan)))+geom_bar(stat="count")
#There are more people without a auto loan

ggplot(credit_wo_woe,aes(x=factor(Presence.of.open.home.loan)))+geom_bar(stat="count")
#There are more people without a home loan


###################### BIVARIATE ANALYSIS ##########################

# *** Demographic data ***

ggplot(demo_wo_woe,aes(x=No.of.dependents,fill=factor(Performance.Tag)))+geom_bar()
#No significant pattern found in defaulter rates for differnt No.of.dependents values.

ggplot(demo_wo_woe,aes(x=Gender,fill=factor(Performance.Tag)))+geom_bar(position="fill")
#No significant pattern found in defaulter rates for gender values

ggplot(demo_wo_woe,aes(x=Profession,fill=factor(Performance.Tag)))+geom_bar(position="fill")
#No significant pattern found in defaulter rates for differnt professions

ggplot(demo_wo_woe,aes(x=Type.of.residence,fill=factor(Performance.Tag)))+geom_bar(position="fill")
#No significant pattern found in defaulter rates for differnt residence type except for 'others' type

ggplot(demo_wo_woe,aes(x=Education,fill=factor(Performance.Tag)))+geom_bar(position="fill")
#No significant pattern found in defaulter rates for differnt Education except for 'others' type

ggplot(demo_wo_woe,aes(x=Marital.Status..at.the.time.of.application.,fill=factor(Performance.Tag)))+geom_bar(position="fill")
#No significant pattern found in defaulter rates for differnt marital status

ggplot(demo_wo_woe,aes(y=Age,x=factor(Performance.Tag)))+geom_boxplot()
#There is no difference in median values. 

ggplot(demo_wo_woe,aes(y=Income,x=factor(Performance.Tag)))+geom_boxplot()
#The median values for income of defaulters are lower than that of non defaulters

ggplot(demo_wo_woe,aes(y=No.of.months.in.current.residence,x=factor(Performance.Tag)))+geom_boxplot()
#The median No.of.months.in.current.residence of non defaulters are lower than that of defaulters

ggplot(demo_wo_woe,aes(y=No.of.months.in.current.company,x=factor(Performance.Tag)))+geom_boxplot()
#The median No.of.months.in.current.company of non defaulters are slightly lower than that of defaulters

ggplot(CreditBureau,aes(x=No.of.times.90.DPD.or.worse.in.last.6.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No of defaulters are increasing with increase in no.of.times 90 DPD in last 6 months

ggplot(CreditBureau,aes(x=No.of.times.60.DPD.or.worse.in.last.6.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No of defaulters are increasing for upto 3  times 60 DPD in last 6 months

ggplot(CreditBureau,aes(x=No.of.times.30.DPD.or.worse.in.last.6.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No of defaulters are increasing for upto 5 times.30 D.P.D in last 6 months

ggplot(CreditBureau,aes(x=No.of.times.90.DPD.or.worse.in.last.12.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No of defaulters are increasing with increase in no.of.times.90.DPD.or worse in last 12 months

ggplot(CreditBureau,aes(x=No.of.times.60.DPD.or.worse.in.last.12.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No of defaulters are increasing till 4 times.60.DPD or worse in last 12 months

ggplot(CreditBureau,aes(x=No.of.times.30.DPD.or.worse.in.last.12.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# # No of defaulters are increasing till 7.5 times.30.DPD in last 12 months

#Binning of values for Modified CC ultilization
credit_wo_woe$Binned_CC_Utilization <- as.factor(cut(credit_wo_woe$Avgas.CC.Utilization.in.last.12.months, breaks = c(0,10, 20, 30, 40, 50, 60,70,80,90,100,110,120),include.lowest = TRUE))
ggplot(credit_wo_woe,aes(x=Binned_CC_Utilization,fill=factor(Performance.Tag)))+geom_bar(position="fill")
#There is no significant pattern found

ggplot(credit_wo_woe,aes(x=No.of.trades.opened.in.last.6.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# Trades which are open 4 times in last 6 months tends to default more

ggplot(CreditBureau,aes(x=No.of.trades.opened.in.last.12.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No appropriate pattern found in no of defaulters with increase in No.of.trades.opened.in.last.12.months

ggplot(CreditBureau,aes(x=No.of.PL.trades.opened.in.last.6.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No of defaulters increases till 4th month then decreases with increase in no of PL trades opened in last 6 months 

ggplot(CreditBureau,aes(x=CreditBureau$No.of.PL.trades.opened.in.last.12.months,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No of defaulters increases till 6th month then decreases and suddenly increases in 12th month with increase in no of PL trades opened in last 12 months

ggplot(CreditBureau,aes(x=No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No appropriate pattern found in no of defaulters 

ggplot(CreditBureau,aes(x=No.of.Inquiries.in.last.12.months..excluding.home...auto.loans.,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No appropriate pattern found in no of defaulters 

# Binning outstanding balance
credit_wo_woe$Binning.outstanding.Balance <- as.factor(cut(credit_wo_woe$Outstanding.Balance, breaks = c(0, 1000000, 2000000, 3000000, 4000000, 5000000 ,6000000),include.lowest = TRUE))
ggplot(credit_wo_woe,aes(x=Binning.outstanding.Balance,fill=factor(Performance.Tag)))+geom_bar(position="fill")
ggplot(CreditBureau,aes(x=CreditBureau$Total.No.of.Trades,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# People who opened 39 No of trades tend to defaut more  

ggplot(credit_wo_woe,aes(x=Presence.of.open.auto.loan,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No appropriate pattern found in no of defaulters 

ggplot(credit_wo_woe,aes(x=Presence.of.open.home.loan,fill=factor(Performance.Tag)))+geom_bar(position="fill")
# No appropriate pattern found in no of defaulters 


###################### Correlation plot ##########################

#** Finding Correlation between Numerical Variables 

corr_data <- merge(demographics, CreditBureau, by ="Application.ID",all=F)

#Removing application ID and Performance tag variables before checking correlation
corr_data <- corr_data[,-c(1,7,8,31,32)]

numerics_data <- sapply(corr_data,is.numeric)
data_correlation <- cor(corr_data[,numerics_data])

View(round(data_correlation,2))
# Let's plot this correlation
corrplot(data_correlation, type = "full",tl.pos = "dt",
        method = "circle", tl.cex = 0.5, tl.col = 'black',
        order = "hclust", diag = FALSE)

#__________________________________________

### Conclusions from EDA:

  #1. Demographic variables are not very good predictors of defaulting. Only below 3 variables seems significant.
      #Income
      #No.of.months.in.current.residence
      #No.of.months.in.current.company

  #2. credit bureau dataset has many variables which seems like good predictors of defaulters.
      #No.of.times.90.DPD.or.worse.in.last.6.months
      #No.of.times.60.DPD.or.worse.in.last.6.months
      #No.of.times.30.DPD.or.worse.in.last.6.months
      #No.of.times.90.DPD.or.worse.in.last.12.months
      #No.of.times.60.DPD.or.worse.in.last.12.months
      #No.of.times.30.DPD.or.worse.in.last.12.months
      #No.of.trades.opened.in.last.6.months
      #No.of.PL.trades.opened.in.last.6.months
      #No.of.PL.trades.opened.in.last.12.months

  #3. There is no correlation between numeric variables of demographic dataset.

  #4. Few numeric variables of Credit bureau dataset show strong positive correlation with other variables.
      #The 6 variables - No.of.times 90/60/30 DPD.or.worse.in.last.6/12 months are highy correlated among themselves.
      #No of enquiries in last 6 months/12 months excluding home, auto loan variables are highly correlated.
      #No. of trade opened in 6/12 months,total number of trades, no of PL trades in 6/12 months are correlated.

#-------------------------------------------------------------------------------#
# Outlier treatment
#-------------------------------------------------------------------------------#

summary(demographics$Age) 
levels(factor(demographics$Age))

# Age field has values less than 18 which is invalid. Since the number of erroneous values is less than 1% in Age column,
# we are directly rejecting them from the dataset.
rejected_age <- demographics[which(demographics$Age < 18),]
demographics <- demographics[-which(demographics$Age < 18),]

#Taking backup of the datasets
demo_with_outliers<-demographics
cb_with_outliers<-CreditBureau

# checking for any outliers in the continuous data using quantiles and replacing them with the nearest non-outlier values.

quantile(demographics$Age,seq(0,1,0.01)) 
demographics$Age[which(demographics$Age<27)]<-27

quantile(demographics$No.of.months.in.current.company,seq(0,1,0.01))
demographics$No.of.months.in.current.company[which(demographics$No.of.months.in.current.company>74)]<-74

quantile(CreditBureau$No.of.trades.opened.in.last.12.months,seq(0,1,0.01))
CreditBureau$No.of.trades.opened.in.last.12.months[which(CreditBureau$No.of.trades.opened.in.last.12.months>21)]<-21

quantile(CreditBureau$Total.No.of.Trades,seq(0,1,0.01))
CreditBureau$Total.No.of.Trades[which(CreditBureau$Total.No.of.Trades>31)]<-31

quantile(CreditBureau$No.of.Inquiries.in.last.12.months..excluding.home...auto.loans.,seq(0,1,0.01))
CreditBureau$No.of.Inquiries.in.last.12.months..excluding.home...auto.loans.[which(CreditBureau$No.of.Inquiries.in.last.12.months..excluding.home...auto.loans. >15)]<- 15

quantile(CreditBureau$No.of.PL.trades.opened.in.last.12.months,seq(0,1,0.01))
CreditBureau$No.of.PL.trades.opened.in.last.12.months[which(CreditBureau$No.of.PL.trades.opened.in.last.12.months > 9)]<- 9

quantile(CreditBureau$No.of.times.30.DPD.or.worse.in.last.12.months,seq(0,1,0.01))
CreditBureau$No.of.times.30.DPD.or.worse.in.last.12.months[which(CreditBureau$No.of.times.30.DPD.or.worse.in.last.12.months > 5)]<- 5

quantile(CreditBureau$No.of.times.60.DPD.or.worse.in.last.12.months,seq(0,1,0.01))
CreditBureau$No.of.times.60.DPD.or.worse.in.last.12.months[which(CreditBureau$No.of.times.60.DPD.or.worse.in.last.12.months>2)]<-2

quantile(CreditBureau$No.of.times.90.DPD.or.worse.in.last.12.months,seq(0,1,0.01))
CreditBureau$No.of.times.90.DPD.or.worse.in.last.12.months[which(CreditBureau$No.of.times.90.DPD.or.worse.in.last.12.months>2)]<-2

quantile(CreditBureau$No.of.times.90.DPD.or.worse.in.last.6.months,seq(0,1,0.01))
CreditBureau$No.of.times.90.DPD.or.worse.in.last.6.months[which(CreditBureau$No.of.times.90.DPD.or.worse.in.last.6.months > 1)]<- 1

quantile(CreditBureau$No.of.times.60.DPD.or.worse.in.last.6.months,seq(0,1,0.01))
CreditBureau$No.of.times.60.DPD.or.worse.in.last.6.months[which(CreditBureau$No.of.times.60.DPD.or.worse.in.last.6.months > 2)]<- 2

quantile(CreditBureau$No.of.times.30.DPD.or.worse.in.last.6.months,seq(0,1,0.01))
CreditBureau$No.of.times.30.DPD.or.worse.in.last.6.months[which(CreditBureau$No.of.times.30.DPD.or.worse.in.last.6.months> 2)]<- 2

quantile(CreditBureau$Avgas.CC.Utilization.in.last.12.months,seq(0,1,0.01))
#there are no outliers after WOE imputation

quantile(CreditBureau$No.of.trades.opened.in.last.6.months,seq(0,1,0.01))
#there are no outliers after WOE imputation

quantile(CreditBureau$No.of.PL.trades.opened.in.last.6.months,seq(0,1,0.01))
CreditBureau$No.of.PL.trades.opened.in.last.6.months[which(CreditBureau$No.of.PL.trades.opened.in.last.6.months> 4)]<- 4

quantile(CreditBureau$No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.,seq(0,1,0.01))
CreditBureau$No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.[which(CreditBureau$No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.> 6)]<- 6

# Taking backup of the datasets
credit_bf_model <- CreditBureau
demo_bf_model <- demographics

#################################################################################
#-------------------------------------------------------------------------------#
# Model Building and Evaluation for Demographic data
#-------------------------------------------------------------------------------#
#################################################################################

prop.table(table(demographics$Performance.Tag))
## Only around 4.2% of observations are under default category. 
## The data is highly imbalanced and hence class balancing is to be performed.

#Converting Performance.Tag as factor type.
demographics$Performance.Tag <- as.factor(demographics$Performance.Tag)

#scaling all variables except target variables and application id to the same scale   
demographics[,-c(1,7)]<-data.frame(scale(demographics[,-c(1,7)]))

# Split into test and train datasets
set.seed(100)
split_indices <- sample.split(demographics$Performance.Tag, SplitRatio = 0.70)
train_demo <- demographics[split_indices, ]
test_demo <- demographics[!split_indices, ]

table(train_demo$Performance.Tag)
#   0     1 
# 46797   2062

#balancing the two classes using ROSE package
train_demo_SMOTE <- ROSE(Performance.Tag ~ ., train_demo,seed=1)$data
table(train_demo_SMOTE$Performance.Tag)
#    0    1 
#  24429 24430  

prop.table(table(train_demo_SMOTE$Performance.Tag))
# 0         1 
# 0.49 0.50

train_demo_SMOTE_other <- train_demo_SMOTE

# Removing variable Application.ID and Reverse.Performance.Tag from train data
train_demo_SMOTE_other$Application.ID <- NULL
train_demo_SMOTE_other$Reverse.Performance.Tag <- NULL

# Removing variable Application.ID and Reverse.Performance.Tag from test data
test_demo_other <- test_demo
test_demo_other$Application.ID <- NULL
test_demo_other$Reverse.Performance.Tag <- NULL


#--------------------------------------------------------------
#************* 1. Logistic Regresssion Model   ****************
#--------------------------------------------------------------

#.............. 1.a) Model Building ...............#

model_lr_1 = glm(Performance.Tag ~ ., data = train_demo_SMOTE_other, family = "binomial")
summary(model_lr_1)

# Variable selection using stepwise AIC algorithm for removing insignificant variables 

model_lr_2<- stepAIC(model_lr_1, direction="both")
summary(model_lr_2)
vif(model_lr_2)
#all vif values are less than 2
#AIC:67253

#predictors are:
#Age                               *  
#Income                            ***
#No.of.months.in.current.residence ***
#No.of.months.in.current.company   ***
#woe.Profession.binned             ***
#woe.Type.of.residence.binned      ***
#woe.Education.binned              ***

# All variables have extremly low p values , hence keeping all varibales on that criteria

final_lr_model <- model_lr_2
test_pred_log = predict(final_lr_model, type = "response", newdata = test_demo_other[,-6])
summary(test_pred_log)

#______________________________________________________

#.............. 1.b) Model Evaluation ...............#

pred <- prediction(test_pred_log,test_demo_other$Performance.Tag)
eva_log<-performance(pred,"sens","spec")
evaA_log<-performance(pred,'acc')
plot(evaA_log)

sensitivity <- eva_log@y.values[[1]]
cutoff <- eva_log@alpha.values[[1]]
specificity<- eva_log@x.values[[1]]
accuracy<-evaA_log@y.values[[1]]
plot(cutoff,sensitivity,col="red")
lines(cutoff,specificity,col="green")
lines(cutoff,accuracy,col="blue")
legend("bottomright", legend=c("sensitivity","accuracy","specificity"),
       col=c("red","blue","green"), lty=1:2, cex=0.8)

abline(v =0.5)
matrix<-data.frame(cbind(sensitivity,specificity,accuracy,cutoff))

final_matrix<-matrix[which(matrix$cutoff>0.1&matrix$cutoff<1),]
matrix<-sapply(final_matrix,function(x) round(x,2))
matrix<-data.frame(matrix)
head(matrix[which(matrix$accuracy==matrix$sensitivity&matrix$sensitivity==matrix$specificity),])

#Cutoff = 0.5
#sensitivity=55%
#specificity=55%
#Accuracy=55%
#Thus a logistic regression model based only on demographic data seems to have low performance.

#--------------------------------------------------------------
#******************* 2. Random Forest   ***********************
#--------------------------------------------------------------

#.............. 2.a) Model Building ...............#

random_rf <- randomForest(Performance.Tag ~., data = train_demo_SMOTE_other, proximity = F, do.trace = T, mtry = 5)
rf_pred <- predict(random_rf, test_demo_other, type = "prob")

#_____________________________________________________

#.............. 2.b) Model Evaluation ...............#

# Finding Cutoff for randomforest to assign yes or no
perform_fn_rf <- function(cutoff) 
{
  predicted_response <- as.factor(ifelse(rf_pred[, 2] >= cutoff, "1", "0"))
  conf <- confusionMatrix(predicted_response, test_demo_other$Performance.Tag, positive = "0")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  OUT_rf <- t(as.matrix(c(sens, spec, acc))) 
  colnames(OUT_rf) <- c("sensitivity", "specificity", "accuracy")
  return(OUT_rf)
}

summary(rf_pred[,2])
s = seq(.00,.91,length=100)
OUT_rf = matrix(0,100,3)

# calculating the sensitivity, specificity and Accuracy for different cutoff values
for(i in 1:100)
{
  OUT_rf[i,] = perform_fn_rf(s[i])
} 

# plotting cutoffs

plot(s, OUT_rf[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT_rf[,2],col="darkgreen",lwd=2)
lines(s,OUT_rf[,3],col=4,lwd=2)

box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))
min(abs(OUT_rf[,1]-OUT_rf[,2]))
cutoff_rf <- s[which(abs(OUT_rf[,1]-OUT_rf[,2])<0.30)]
cutoff_rf
#  0.2481818 0.2573737 0.2665657 0.2757576 0.2849495 0.2941414 0.3033333 0.3125253

# The plot shows that cutoff value of around 0.28 optimises sensitivity and accuracy
predicted_response <- factor(ifelse(rf_pred[, 2] >= 0.28, "1", "0"))
conf_forest <- confusionMatrix(predicted_response, test_demo_other$Performance.Tag, positive = "0")
conf_forest

#Confusion Matrix and Statistics

#          Reference
# Prediction     0     1
#       0      10943   416
#       1       9113   468

#Accuracy : 0.5449          
#Sensitivity : 0.54562         
#Specificity : 0.52941         
#'Positive' Class : 0    

#========== Conclusion ==========
# Random forest model perfoms almost equally for demographic data compared to 
# logistic regression model. But the overall performance is very low for models made from demographic data alone.
# Hence moving forward with building models for demographic and credit data combined.


#########################################################################################
#---------------------------------------------------------------------------------------#
# Model Building and Evaluation for merged data (without performance tag missing records)
#---------------------------------------------------------------------------------------#
#########################################################################################

#checking if Performance tag and Application ID field is identical across credit bureau and demographic dataset
setdiff(demographics$Performance.Tag, CreditBureau$Performance.Tag)         
#integer(0) -> Identical Performance tag across these dataset

setdiff(demographics$Application.ID, CreditBureau$Application.ID)         
#integer(0) -> Identical Application ID key field across these dataset

#Since Performance tag and Reverse Performance tag fields are identical accross 2 datasets, one of their occurances can be removed.
CreditBureau$Performance.Tag<-NULL
CreditBureau$Reverse.Performance.Tag<-NULL

master_df <- merge(demographics, CreditBureau, by ="Application.ID",all=F)
dim(master_df)  #[1] 69799 rows and 30 columns

#Duplicate rows in data - none found.
sum(duplicated(master_df))   

#removing columns in merged_data not required for model building
master_df$Reverse.Performance.Tag<-NULL
master_df$Application.ID<-NULL

#scaling the data
master_df[,-6]<-data.frame(scale(master_df[,-6]))

#Splitting data into train and test
set.seed(100)
split_indices <- sample.split(master_df$Performance.Tag, SplitRatio = 0.70)
train_cb <- master_df[split_indices, ]
test_cb <- master_df[!split_indices, ]

table(train_cb$Performance.Tag)
#   0     1 
# 46797   2062

#The data is highly imbalanced. Using ROSE sampling technique to balance the data.
train_cb_SMOTE <- ROSE(Performance.Tag ~ ., train_cb,seed=1)$data
table(train_cb_SMOTE$Performance.Tag)
#    0    1 
#  24429 24430  

prop.table(table(train_cb_SMOTE$Performance.Tag))
# 0         1 
# 0.499 0.50

######## Note  :- Values for evaluation metrics for different models may vary slightly as ###############
######## we have used the command 'registerDOseq()'for parallel computing as otherwise ##################
######## we get the error "error in summary.connection " for models like SVM and Random Forest ##########


#--------------------------------------------------------------
#************* 1. Logistic Regresssion Model   ****************
#--------------------------------------------------------------

#.............. 1.a) Model Building ...............#

model_lg_1 = glm(Performance.Tag ~ ., data = train_cb_SMOTE, family = "binomial")
summary(model_lg_1)

# Variable selection using stepwise AIC algorithm for removing insignificant variables 

model_lg_2<- stepAIC(model_lg_1, direction="both")
summary(model_lg_2)
vif(model_lg_2)

#AIC:63942

#removing Age variable with lesser p value
model_lg_3<-glm(formula = Performance.Tag ~ Income + No.of.months.in.current.residence + 
                   No.of.months.in.current.company + woe.Education.binned + 
                   woe.Marital.Status..at.the.time.of.application..binned + 
                   No.of.times.90.DPD.or.worse.in.last.6.months + No.of.times.60.DPD.or.worse.in.last.6.months + 
                   No.of.times.30.DPD.or.worse.in.last.6.months + No.of.times.90.DPD.or.worse.in.last.12.months + 
                   No.of.times.30.DPD.or.worse.in.last.12.months + Avgas.CC.Utilization.in.last.12.months + 
                   No.of.trades.opened.in.last.6.months + No.of.PL.trades.opened.in.last.6.months + 
                   No.of.PL.trades.opened.in.last.12.months + No.of.Inquiries.in.last.12.months..excluding.home...auto.loans. + 
                   Presence.of.open.home.loan + Outstanding.Balance + Total.No.of.Trades, 
                 family = "binomial", data = train_cb_SMOTE)
summary(model_lg_3)
#AIC: 63943

#removing woe.Marital.Status..at.the.time.of.application..binned  variable with lesser p value
model_lg_4<-glm(formula = Performance.Tag ~ Income + No.of.months.in.current.residence + 
                   No.of.months.in.current.company + woe.Education.binned +
                   No.of.times.90.DPD.or.worse.in.last.6.months + No.of.times.60.DPD.or.worse.in.last.6.months + 
                   No.of.times.30.DPD.or.worse.in.last.6.months + No.of.times.90.DPD.or.worse.in.last.12.months + 
                   No.of.times.30.DPD.or.worse.in.last.12.months + Avgas.CC.Utilization.in.last.12.months + 
                   No.of.trades.opened.in.last.6.months + No.of.PL.trades.opened.in.last.6.months + 
                   No.of.PL.trades.opened.in.last.12.months + No.of.Inquiries.in.last.12.months..excluding.home...auto.loans. + 
                   Presence.of.open.home.loan + Outstanding.Balance + Total.No.of.Trades, 
                 family = "binomial", data = train_cb_SMOTE)
summary(model_lg_4)
vif(model_lg_4)
#AIC: 63945
#All variable are significant and vif values less than or almost equal to 2

merged_log_model <- model_lg_4

#_____________________________________________________

#.............. 1.b) Model Evaluation ...............#

#### 1.b.1) Confusion matrix:

test_pred_log = predict(merged_log_model, type = "response",newdata = test_cb[,-6])
summary(test_pred_log)

pred <- prediction(test_pred_log,test_cb$Performance.Tag)
eva_log<-performance(pred,"sens","spec")
evaA_log<-performance(pred,'acc')
plot(evaA_log)

sensitivity <- eva_log@y.values[[1]]
cutoff <- eva_log@alpha.values[[1]]
specificity<- eva_log@x.values[[1]]
accuracy<-evaA_log@y.values[[1]]
plot(cutoff,sensitivity,col="red")
lines(cutoff,specificity,col="green")
lines(cutoff,accuracy,col="blue")
legend("bottomright", legend=c("sensitivity","accuracy","specificity"),
       col=c("red","blue","green"), lty=1:2, cex=0.8)

abline(v =0.51)
matrix<-data.frame(cbind(sensitivity,specificity,accuracy,cutoff))

final_matrix<-matrix[which(matrix$cutoff>0.1&matrix$cutoff<1),]
matrix<-sapply(final_matrix,function(x) round(x,2))
matrix<-data.frame(matrix)
head(matrix[which(matrix$accuracy==matrix$sensitivity&matrix$sensitivity==matrix$specificity),])

#choosing a final cutoff value of 0.51
#sensitivity=63%
#specificity=63%
#accuracy=63%

test_cutoff <- factor(ifelse(test_pred_log >=0.51, "1", "0"))
test_actual <- factor(ifelse(test_cb$Performance.Tag==1, "1", "0"))
conf_final <- confusionMatrix(test_cutoff, test_actual, positive = "0")
conf_final
#           Reference
#Prediction     0     1
#0            12436   324
#1             7620   560


#### 1.b.2) KS Statistics:

perf<-performance(pred,'tpr','fpr')
ks<-max(perf@y.values[[1]]-perf@x.values[[1]])
ks
# ks statistic for this model is 0.27


#### 1.b.3) Gain and Lift charts:
gain<-performance(pred,'tpr','rpp')
deciles<-performance(pred,'rpp')

#ks chart
k_stat_matrix<-data.frame(10*(deciles@y.values[[1]]),(perf@y.values[[1]]-perf@x.values[[1]]))
colnames(k_stat_matrix)[1]<-"deciles"
colnames(k_stat_matrix)[2]<-"k_statistic"
k_stat_matrix$k_statistic<-round(k_stat_matrix$k_statistic,2)

plot(k_stat_matrix)
abline(h=2.7,v=5)
#ks statistic lies withinin first 5 deciles 

plot(gain)
#within first 4 deciles as per the model we are able to predict 60% of defaulters correctly

plot(perf)
plotLift(test_pred_log,test_cb$Performance.Tag)
#a lift of 1.6 times is achieved with the model within first 4 deciles compared to random model

#### 1.b.4) Area under the curve:
plot(roc(test_pred_log,factor(test_cb$Performance.Tag)))
auc(roc(test_pred_log,factor(test_cb$Performance.Tag)))
# area under ROC CURVE is 0.67

##############################################################
#Cross validation on other test data set:
set.seed(50)
split_indices <- sample.split(master_df$Performance.Tag, SplitRatio = 0.70)

test_cb <- master_df[!split_indices, ]
test_pred_log = predict(merged_log_model, type = "response", newdata = test_cb[,-6])

#Lets find cofusion matrix for this model
test_cutoff <- factor(ifelse(test_pred_log >=0.51, "1", "0"))
test_actual <- factor(ifelse(test_cb$Performance.Tag==1, "1", "0"))
conf_final <- confusionMatrix(test_cutoff, test_actual, positive = "0")
conf_final
#sensitivity=62%
#specificity=63%
#accuracy=62%

##############################################################
#Cross validation on other test data set:
set.seed(25)
split_indices <- sample.split(master_df$Performance.Tag, SplitRatio = 0.70)

test_cb <- master_df[!split_indices, ]
test_pred_log = predict(merged_log_model, type = "response",  newdata = test_cb[,-6])

#Lets find cofusion matrix for this model
test_cutoff <- factor(ifelse(test_pred_log >=0.51, "1", "0"))
test_actual <- factor(ifelse(test_cb$Performance.Tag==1, "1", "0"))
conf_final <- confusionMatrix(test_cutoff, test_actual, positive = "0")
conf_final
#sensitivity=61%
#specificity=62%
#accuracy=61%

#conclusion: Above 60% accuracy,sensitivity and specificity
#was achieved on different test data sets with a logistic regression model on 
#combined demographic and merged data.

#--------------------------------------------------------------
#****************** 2. XGBoost Model   ************************
#--------------------------------------------------------------

#.............. 2.a) Model Building ...............#

dummy<-master_df
master_df$Performance.Tag<-as.character(dummy$Performance.Tag)
master_df$Performance.Tag<-as.numeric(master_df$Performance.Tag)

set.seed(100)
split_indices <- sample.split(master_df$Performance.Tag, SplitRatio = 0.70)
train <- master_df[split_indices, ]
test <- master_df[!split_indices, ]

#The data is highly imbalanced. Using ROSE sampling technique to balance the data.
table(train$Performance.Tag)
#   0     1 
# 46797   2062

#balancing the two classes using ROSE package

train <- ROSE(Performance.Tag ~ ., train,seed=1)$data
table(train$Performance.Tag)

train = as.matrix(sapply(train,as.numeric))
test = as.matrix(sapply(test,as.numeric))
final<-as.matrix(sapply(master_df,as.numeric))

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric ="auc",
  eta=0.7,
  gamma=5,
  max_depth=3,
  min_child_weight=2,
  subsample=1,
  colsample_bytree= 0.933
)

#.............. 2.b) Model Evaluation ...............#

#xgboost cross validation
bst.cv = xgb.cv(
  params=params,
  data = train[,-6],
  label =train[,6],
  nfold = 5,
  nrounds=300,
  prediction=T)
#highest test AUC was obtained was 0.7556 after 193 rounds of 5 fold cross validation

watchlist <- list(train, test)
xgb.model <- xgboost(params=params, data=train[,-6],
                     label=train[,6],
                     nrounds = 193, watchlist)

#### 2.b.1) Confusion matrix:

xgb.predictions <- predict(xgb.model, test[,-6],type = "response")

true<-data.frame(test)
test_pred<-data.frame(xgb.predictions)
pred<-prediction(test_pred,true$Performance.Tag)
eva<-performance(pred,"sens","spec")
evaA<-performance(pred,'acc')

plot(evaA)
sensitivity <- eva@y.values[[1]]
cutoff <- eva@alpha.values[[1]]
specificity<- eva@x.values[[1]]
accuracy<-evaA@y.values[[1]]
plot(cutoff,sensitivity,col="red")
lines(cutoff,specificity,col="green")
lines(cutoff,accuracy,col="blue")
abline(v=0.2)
legend("bottomright", legend=c("sensitivity","accuracy","specificity"),
       col=c("red","blue","green"), lty=1:2, cex=0.6)
matrix<-data.frame(cbind(sensitivity,specificity,accuracy,cutoff))

final_matrix<-matrix[which(matrix$cutoff>0.1&matrix$cutoff<1),]
matrix<-sapply(final_matrix,function(x) round(x,4))
matrix<-data.frame(matrix)
head(matrix[which(matrix$accuracy==matrix$sensitivity&matrix$sensitivity==matrix$specificity),])

#Lets find cofusion matrix for this model
test_cutoff <- factor(ifelse(xgb.predictions >=0.195, "1", "0"))
test_actual <- factor(ifelse(true$Performance.Tag==1, "1", "0"))
conf_final <- confusionMatrix(test_cutoff , test_actual, positive = "0")
conf_final

#Accuracy : 0.6413 
#Specificity : 0.6413   
#Sensitivity : 0.6414

#           Reference
#Prediction     0     1
#        0  12862   317
#        1   7194   567


#### 2.b.2) KS Statistics:
perf<-performance(pred,'tpr','fpr')
ks<-max(perf@y.values[[1]]-perf@x.values[[1]])
ks
# ks statistic for this model is 0.28

gain<-performance(pred,'tpr','rpp')
deciles<-performance(pred,'rpp')

#ks chart
k_stat_matrix<-data.frame(10*(deciles@y.values[[1]]),(perf@y.values[[1]])-perf@x.values[[1]])
colnames(k_stat_matrix)[1]<-"deciles"
colnames(k_stat_matrix)[2]<-"k_statistic"
k_stat_matrix$k_statistic<-round(k_stat_matrix$k_statistic,2)

plot(k_stat_matrix)
abline(v=4)
#ks statistic lies within first 4 deciles
###gain and lift chart

#### 2.b.3) Gain and Lift charts:

plot(gain)
abline(h=0.75,v=0.49)
#within first 5 deciles as per the model we are able to capture 75% of defaulters correctly.

plot(perf)
plotLift(xgb.predictions,true$Performance.Tag)
abline(h=1.8,v=2.9)
#a lift little below than 1.8 times is achieved with the model compared to a random model within first 3 deciles


#### 2.b.4) Area under the curve:
plot(roc(xgb.predictions,factor(true$Performance.Tag)))
auc(roc(xgb.predictions,factor(true$Performance.Tag)))
 #0.67

xgb.importance(model=xgb.model)
#Variable importance in terms of gain(fractional contribution of each feature to the model)
#1:                    Avgas.CC.Utilization.in.last.12.months
#2:                    No.of.times.30.DPD.or.worse.in.last.12.months
#3:                    No.of.times.90.DPD.or.worse.in.last.6.months
#4:                    No.of.times.60.DPD.or.worse.in.last.6.months
#5:                    No.of.times.30.DPD.or.worse.in.last.6.months

#--------------------------------------------------------------
#************* 3. Support vector machines (SVM)   *************
#--------------------------------------------------------------

train<-train_cb_SMOTE
test<-test_cb

set.seed(50)
split_indices <- sample.split(train$Performance.Tag, SplitRatio = 0.05)

train_svm <- train[split_indices, ]
prop.table(table(train_svm$Performance.Tag))

#.............. 3.a) Model Building ...............#

######### Linear Kernal ########

linear_model <- ksvm(Performance.Tag~ ., data = train_svm, scale = FALSE, kernel = "vanilladot")
linear_eval <- predict(linear_model, test_cb[,-6])

linear_conf <- confusionMatrix(linear_eval,test_cb$Performance.Tag)
linear_conf
#a very low accuracy of 0.53 is obtained with a linear kernel

######### RBF Kernal ########

RBF_model <- ksvm(Performance.Tag~ ., data = train_svm, scale = FALSE, kernel = "rbfdot")
RBF_eval <- predict(RBF_model, test_cb[,-6])

RBF_conf <- confusionMatrix(RBF_eval,test_cb$Performance.Tag)
RBF_conf
#Accuracy : 67%
#Sensitivity : 68%
#Specificity : 56%


#.............. 3.b) Model Evaluation ...............#

#### b.3 Hyperparameter tuning and cross validation

trainControl <- trainControl(method="cv", number=2)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.
metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
RBF_model

#cost C = 1 
#sigma =  0.022
#Training error : 0.299222
#Number of Support Vectors : 2043 

registerDoSEQ()
grid <- expand.grid( C=c(1,2,3),sigma=c(0.01,0.02))

fit.svm_rbf <- train(Performance.Tag~., data=train_svm, method="svmRadial", metric="Accuracy", 
                     tuneGrid=grid, trControl=trainControl)
print(fit.svm_rbf)
plot(fit.svm_rbf)

#C  sigma  Accuracy   Kappa    
#1  0.01   0.61       0.23
#1  0.02   0.62       0.25
#2  0.01   0.62       0.24
#2  0.02   0.62       0.24
#3  0.01   0.62       0.25
#3  0.02   0.62       0.25

#Accuracy was used to select the optimal model using
#the largest value.
#The final values used for the model were sigma = 0.02 and
#C = 1.
##################################################################

#### a.4 Valdiating the model after cross validation on test data

evaluate_rbf <- predict(fit.svm_rbf, test_cb[,-6])

confusionMatrix(evaluate_rbf, test_cb$Performance.Tag)
# Accuracy : 0.67
# Sensitivity : 0.67        
#Specificity : 0.57

#--------------------------------------------------------------
#*************** 4. Random Forest model   ********************
#--------------------------------------------------------------

#.............. 4.a) Model Building ...............#

merged_rf <- randomForest(Performance.Tag ~., data = train_cb_SMOTE, proximity = F, do.trace = T, ntree=350, mtry = 8)
mer_rf_pred <- predict(merged_rf, test_cb, type = "prob")

# Cutoff for randomforest to assign yes or no
perform_fn_rf_mer <- function(cutoff) 
{
  predicted_response_mer <- as.factor(ifelse(mer_rf_pred[, 2] >= cutoff, "1", "0"))
  conf <- confusionMatrix(predicted_response_mer, test_cb$Performance.Tag, positive = "0")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  OUT_rf_mer <- t(as.matrix(c(sens, spec, acc))) 
  colnames(OUT_rf_mer) <- c("sensitivity", "specificity", "accuracy")
  return(OUT_rf_mer)
}

#.............. 4.b) Model Evaluation ...............#

summary(mer_rf_pred[,2])
s = seq(.00,.91,length=100)
OUT_rf_mer = matrix(0,100,3)

# calculate the sens, spec and acc for different cutoff values
for(i in 1:100)
{
  OUT_rf_mer[i,] = perform_fn_rf_mer(s[i])
} 

# plotting cutoffs
plot(s, OUT_rf_mer[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT_rf_mer[,2],col="darkgreen",lwd=2)
lines(s,OUT_rf_mer[,3],col=4,lwd=2)
abline(v =0.17)

box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))
min(abs(OUT_rf_mer[,1]-OUT_rf_mer[,2]))
cutoff_rf_mer <- s[which(abs(OUT_rf_mer[,1]-OUT_rf_mer[,2])<0.25)]
cutoff_rf_mer

#[1] 0.1011111 0.1103030 0.1194949 0.1286869 0.1378788 0.1470707 0.1562626 0.1654545 0.1746465 0.1838384 0.1930303 0.2022222
#[13] 0.2114141 0.2206061 0.2297980 0.2389899 0.2481818 0.2573737 0.2665657 0.2757576 0.2849495 0.2941414 0.3033333

# choosing the cutoff value of 0.17
predicted_response_mer <- factor(ifelse(mer_rf_pred[, 2] >= 0.17, "1", "0"))
conf_forest_mer <- confusionMatrix(predicted_response_mer, test_cb$Performance.Tag, positive = "0")
conf_forest_mer

#Confusion Matrix and Statistics

#         Reference
#Prediction     0     1
#         0   12874   329
#         1   7182   555

# Accuracy : 0.64          
#Sensitivity : 0.64
#Specificity : 0.63
  
##########   cross validation  of Random Forest

control <- trainControl(method="cv", number=5)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- (8:10)
tunegrid <- expand.grid(.mtry=mtry)

rf_default_mer <- train(Performance.Tag~., data=train_cb_SMOTE, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default_mer)
  
#Random Forest 

#48859 samples
#27 predictor
#2 classes: '0', '1' 

#Resampling: Cross-Validated (5 fold) 
#Summary of sample sizes: 39087, 39087, 39087, 39088, 39087 
#Resampling results across tuning parameters:
  
#  mtry  Accuracy    Kappa    
#   8    0.692       0.385
#   9    0.691       0.382
#  10    0.692       0.384

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 8.

plot(rf_default_mer)
evaluate_model_cv <- predict(rf_default_mer, test_cb, type = "prob")

perform_fn_rf_cv <- function(cutoff) 
{
  predicted_response_cv <- as.factor(ifelse(evaluate_model_cv[, 2] >= cutoff, "1", "0"))
  conf <- confusionMatrix(predicted_response_cv, test_cb$Performance.Tag, positive = "0")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  OUT_rf_cv <- t(as.matrix(c(sens, spec, acc))) 
  colnames(OUT_rf_cv) <- c("sensitivity", "specificity", "accuracy")
  return(OUT_rf_cv
  )
}

summary(evaluate_model_cv[,2])
s1 = seq(.00,.91,length=100)
OUT_rf_cv = matrix(0,100,3)

# calculate the sens, spec and acc for different cutoff values
for(i in 1:100)
{
  OUT_rf_cv[i,] = perform_fn_rf_cv(s1[i])
} 

plot(s, OUT_rf_cv[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s1,OUT_rf_cv[,2],col="darkgreen",lwd=2)
lines(s1,OUT_rf_cv[,3],col=4,lwd=2)
abline(v=0.17)

box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))
min(abs(OUT_rf_cv[,1]-OUT_rf_cv[,2]))
cutoff_rf_cv <- s[which(abs(OUT_rf_cv[,1]-OUT_rf_cv[,2])<0.25)]
cutoff_rf_cv

# [1] 0.1103030 0.1194949 0.1286869 0.1378788 0.1470707 0.1562626 0.1654545 0.1746465 0.1838384 0.1930303 0.2022222 0.2114141
# [13] 0.2206061 0.2297980 0.2389899 0.2481818 0.2573737 0.2665657 0.2757576 0.2849495 0.2941414 0.3033333

predicted_response_cv <- factor(ifelse(evaluate_model_cv[, 2] >= 0.17, "1", "0"))
conf_forest_cv <- confusionMatrix(predicted_response_cv, test_cb$Performance.Tag, positive = "0")
conf_forest_cv

#Confusion Matrix and Statistics

#Reference
# Prediction     0     1
#          0  12726   321
#          1  7330   563

#Accuracy : 0.634
#Sensitivity : 0.634       
#Specificity : 0.636


#========== Conclusion ==========

# All the 4 models built on merged dataset produces accuracy of around 63% - 64%. 
# Hence considering logistic model for predicting performance tag missing values

######## Note : We have trained the model on the available performance tags and           ########
######## predicted the missing values of performance tag variable using predictive models ########
######## Now adding the predicted values to the merged dataset to train the model again   ########


#################################################################################
#-------------------------------------------------------------------------------#
# Data preparation for datasets including performance tag missing records
#-------------------------------------------------------------------------------#
#################################################################################
  
#********* Cleansing Rejected dataset ************#

setdiff(rejected_demo$Application.ID, rejected_credit$Application.ID) 
#Since Performance tag and Reverse Performance tag fields are identical accross 2 datasets, one of their occurances can be removed.

# ************* WOE and IV for Demographic data *************

#making a dummy new variable 'Reverse.Performance.Tag'for sake of woe imputation.
rejected_demo$Reverse.Performance.Tag <- ifelse(rejected_demo$Performance.Tag == 0,1,0)
## creating woe buckets for numerical variables

#making a dataset containing character variables with missing values along with "Reverse.Performance.Tag"
colnames <- c("Reverse.Performance.Tag","Marital.Status..at.the.time.of.application.","Gender","Education","Profession","Type.of.residence")
char_missing <- rejected_demo[,colnames(rejected_demo)%in%colnames]
df.with.binned.vars.added <- woe.binning.deploy(char_missing, binning,add.woe.or.dum.var='woe')
demo_woe_rejected <-df.with.binned.vars.added[,c(8,10,12,14,16)]

#replacing 5 char variables having missing values with their respective WOE values.
colnames <- c("Marital.Status..at.the.time.of.application.","Gender","Education","Profession","Type.of.residence")
rejected_demo <- rejected_demo[,!colnames(rejected_demo)%in%colnames]
rejected_demo <- cbind(rejected_demo,demo_woe_rejected)

rejected_demo$No.of.dependents[which(is.na(rejected_demo$No.of.dependents))]<-0
for(i in 1:nrow(rejected_demo))
{
  
  if(rejected_demo$No.of.dependents[i]>=1 & rejected_demo$No.of.dependents[i]< 4)
  {
    if(rejected_demo$No.of.dependents[i]<=3)
    {rejected_demo$No.of.dependents[i]<- -0.005}
  }
  else if (rejected_demo$No.of.dependents[i]>= 4)
  {rejected_demo$No.of.dependents[i]<- 0.01}
  else{rejected_demo$No.of.dependents[i]<- 0}
}

summary(rejected_demo$No.of.dependents)   
rejected_demo$Reverse.Performance.Tag <-NULL
#-------------------------------------------------------------------

# ************* Woe and IV for Credit Bureau data *************

#making a dummy new variable 'Reverse.Performance.Tag'for sake of woe imputation.
rejected_credit$Reverse.Performance.Tag <- ifelse(rejected_credit$Performance.Tag == 0,1,0)

#binning "Avgas.CC.Utilization.in.last.12.months"
colnames<- c("Reverse.Performance.Tag","Avgas.CC.Utilization.in.last.12.months")
char_missing <- rejected_credit[,names(rejected_credit)%in%colnames]
char_missing <- woe.binning.deploy(char_missing, binning_Avgas.CC.Utilization.in.last.12.months,add.woe.or.dum.var='woe')
rejected_credit$Avgas.CC.Utilization.in.last.12.months<-char_missing$woe.Avgas.CC.Utilization.in.last.12.months.binned

#binning all other  variables with missing values in Credit Bureau data
colnames<- c("Presence.of.open.home.loan","Reverse.Performance.Tag","No.of.trades.opened.in.last.6.months","Outstanding.Balance")
char_missing <- rejected_credit[,names(rejected_credit)%in%colnames]
char_missing <- woe.binning.deploy(char_missing, binning_b,add.woe.or.dum.var='woe')

#replacing all other missing variables in Credit Bureau data with their WOE values
rejected_credit$No.of.trades.opened.in.last.6.months<-char_missing$woe.No.of.trades.opened.in.last.6.months.binned
rejected_credit$Presence.of.open.home.loan<-char_missing$woe.Presence.of.open.home.loan.binned
rejected_credit$Outstanding.Balance<-char_missing$woe.Outstanding.Balance.binned

rejected_credit$Reverse.Performance.Tag <-NULL
rejected_credit$Performance.Tag <-NULL
colSums(is.na(rejected_credit))
#0
colSums(is.na(rejected_demo))
#1425 performance tag

########################################################
# Merging the Performace tag rejected records to merged dataset

master_rejected<- merge(rejected_demo, rejected_credit, by ="Application.ID",all=F)
master_rejected$Application.ID<-NULL
cb_with_outliers$Performance.Tag<-NULL
cb_with_outliers$Reverse.Performance.Tag<-NULL
demo_with_outliers$Reverse.Performance.Tag<-NULL

master_non_rejected<-merge(demo_with_outliers, cb_with_outliers, by ="Application.ID",all=F)
master_non_rejected$Application.ID<-NULL
master<-rbind(master_non_rejected,master_rejected)

##########################################################
#CHECKING FOR OUTLIERS IN Performance tag missing records 

master[which(master$Age < 18),]

quantile(master$Age,seq(0,1,0.01)) 
master$Age[which(master$Age<27)]<-27

quantile(master$No.of.months.in.current.company,seq(0,1,0.01))
master$No.of.months.in.current.company[which(master$No.of.months.in.current.company>74)]<-74

quantile(master$No.of.trades.opened.in.last.12.months,seq(0,1,0.01))
master$No.of.trades.opened.in.last.12.months[which(master$No.of.trades.opened.in.last.12.months>21)]<-21

quantile(master$Total.No.of.Trades,seq(0,1,0.01))
master$Total.No.of.Trades[which(master$Total.No.of.Trades>31)]<-31

quantile(master$No.of.Inquiries.in.last.12.months..excluding.home...auto.loans.,seq(0,1,0.01))
master$No.of.Inquiries.in.last.12.months..excluding.home...auto.loans.[which(master$No.of.Inquiries.in.last.12.months..excluding.home...auto.loans. >15)]<- 15

quantile(master$No.of.PL.trades.opened.in.last.12.months,seq(0,1,0.01))
master$No.of.PL.trades.opened.in.last.12.months[which(master$No.of.PL.trades.opened.in.last.12.months > 9)]<- 9

quantile(master$No.of.times.30.DPD.or.worse.in.last.12.months,seq(0,1,0.01))
master$No.of.times.30.DPD.or.worse.in.last.12.months[which(master$No.of.times.30.DPD.or.worse.in.last.12.months > 5)]<- 5

quantile(master$No.of.times.60.DPD.or.worse.in.last.12.months,seq(0,1,0.01))
master$No.of.times.60.DPD.or.worse.in.last.12.months[which(master$No.of.times.60.DPD.or.worse.in.last.12.months>2)]<-2

quantile(master$No.of.times.90.DPD.or.worse.in.last.12.months,seq(0,1,0.01))
master$No.of.times.90.DPD.or.worse.in.last.12.months[which(master$No.of.times.90.DPD.or.worse.in.last.12.months>2)]<-2

quantile(master$No.of.times.90.DPD.or.worse.in.last.6.months,seq(0,1,0.01))
master$No.of.times.90.DPD.or.worse.in.last.6.months[which(master$No.of.times.90.DPD.or.worse.in.last.6.months > 1)]<- 1

quantile(master$No.of.times.60.DPD.or.worse.in.last.6.months,seq(0,1,0.01))
master$No.of.times.60.DPD.or.worse.in.last.6.months[which(master$No.of.times.60.DPD.or.worse.in.last.6.months > 2)]<- 2

quantile(master$No.of.times.30.DPD.or.worse.in.last.6.months,seq(0,1,0.01))
master$No.of.times.30.DPD.or.worse.in.last.6.months[which(master$No.of.times.30.DPD.or.worse.in.last.6.months> 2)]<- 2

quantile(master$No.of.PL.trades.opened.in.last.6.months,seq(0,1,0.01))
master$No.of.PL.trades.opened.in.last.6.months[which(master$No.of.PL.trades.opened.in.last.6.months> 4)]<- 4

quantile(master$No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.,seq(0,1,0.01))
master$No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.[which(master$No.of.Inquiries.in.last.6.months..excluding.home...auto.loans.> 6)]<- 6

####################################################
# Data Scaling:

master[,-6]<-data.frame(scale(master[,-6]))
no_tags<-master[c(69800:71224),]
master<-master[-c(69800:71224),]

#using our logistic regression model to make predictions for rejected applications
pred_no_tags = predict(merged_log_model, type = "response",newdata = no_tags[,-6])
summary(pred_no_tags)

no_tags$Performance.Tag <- as.numeric(ifelse(pred_no_tags >=0.51, "1", "0"))
master[c(69800:71224),]<-no_tags
summary(factor(no_tags$Performance.Tag))
#0    1 
#9 1416
#This indicates most of the rejected applicants are correctly predicted to be defaulters.

master$Performance.Tag<-factor(master$Performance.Tag)

###################################################
#Treating class imbalance

# Split into test and train datasets
set.seed(100)
split_indices <- sample.split(master$Performance.Tag, SplitRatio = 0.70)
train <- master[split_indices, ]
test <- master[!split_indices, ]

table(train$Performance.Tag)
#   0     1 
# 46803   3053

#balancing the two classes using ROSE package
train_SMOTE <- ROSE(Performance.Tag ~ ., train,seed=1)$data
table(train_SMOTE$Performance.Tag)
#    0    1 
#  24917 24939  

prop.table(table(train_SMOTE$Performance.Tag))
# 0         1 
# 0.49 0.50

#-------------------------------------------------------------------------------#
# ****** 1. Random Forest (Non-rejected + Missing performance tag records) *****
#-------------------------------------------------------------------------------#

#.............. 1.a) Model Building ...............#

train_rf <- randomForest(Performance.Tag ~., data = train_SMOTE, proximity = F, do.trace = T, ntree=500, mtry = 5)
train_SMOTE_pred <- predict(train_rf, test, type = "prob")

#_______________________________________________________

#.............. 1.b) Model Evaluation ...............#

perform_fun_rf_train <- function(cutoff) 
{
  predicted_response_train <- as.factor(ifelse(train_SMOTE_pred[, 2] >= cutoff, "1", "0"))
  conf <- confusionMatrix(predicted_response_train, test$Performance.Tag, positive = "0")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  OUT_rf_train <- t(as.matrix(c(sens, spec, acc))) 
  colnames(OUT_rf_train) <- c("sensitivity", "specificity", "accuracy")
  return(OUT_rf_train)
}

summary(train_SMOTE_pred[,2])
s2 = seq(.00,.91,length=100)
OUT_rf_train = matrix(0,100,3)

# calculate the sens, spec and acc for different cutoff values
for(i in 1:100)
{
  OUT_rf_train[i,] = perform_fun_rf_train(s2[i])
} 

plot(s2, OUT_rf_train[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s2,OUT_rf_train[,2],col="darkgreen",lwd=2)
lines(s2,OUT_rf_train[,3],col=4,lwd=2)
legend("bottomright", legend=c("sensitivity","accuracy","specificity"),
       col=c("red","blue","green"), lty=1:2, cex=0.8)
abline(v=0.32)

box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))
min(abs(OUT_rf_train[,1]-OUT_rf_train[,2]))
cutoff_rf_train <- s2[which(abs(OUT_rf_train[,1]-OUT_rf_train[,2])<0.50)]
cutoff_rf_train

#[1] 0.06434343 0.07353535 0.08272727 0.09191919 0.10111111 0.11030303 0.11949495 0.12868687 0.13787879 0.14707071 0.15626263
#[12] 0.16545455 0.17464646 0.18383838 0.19303030 0.20222222 0.21141414 0.22060606 0.22979798 0.23898990 0.24818182 0.25737374
#[23] 0.26656566 0.27575758 0.28494949 0.29414141 0.30333333 0.31252525 0.32171717 0.33090909 0.34010101 0.34929293 0.35848485
#[34] 0.36767677 0.37686869 0.38606061 0.39525253 0.40444444 0.41363636 0.42282828 0.43202020 0.44121212 0.45040404 0.45959596
#[45] 0.46878788 0.47797980 0.48717172 0.49636364 0.50555556 0.51474747 0.52393939 0.53313131 0.54232323 0.55151515 0.56070707
#[56] 0.56989899 0.57909091 0.58828283 0.59747475 0.60666667 0.61585859 0.62505051 0.63424242 0.64343434 0.65262626 0.66181818
#[67] 0.67101010 0.68020202 0.68939394 0.69858586 0.70777778 0.71696970 0.72616162

# Choosing cutoff at value of 0.32

predicted_response_train <- factor(ifelse(train_SMOTE_pred[, 2] >= 0.32, "1", "0"))
conf_forest_train <- confusionMatrix(predicted_response_train, test$Performance.Tag, positive = "0")
conf_forest_train

# Confusion Matrix and Statistics

         #Reference
# Prediction     0     1
#          0  14045   404
#          1   6014   905

# Accuracy : 0.70          
#Kappa : 0.13          
#Sensitivity : 0.70
#Specificity : 0.69

#_______________________________________________________

#***Cross Validation of Random Forest Model***

control <- trainControl(method="cv", number=5)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- (7:11)
tunegrid <- expand.grid(.mtry=mtry)

rf_default_train_SMOTE <- train(Performance.Tag~., data=train_SMOTE, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default_train_SMOTE)

# 49856 samples
# 27 predictor
# 2 classes: '0', '1' 

# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 39885, 39885, 39884, 39885, 39885 
# Resampling results across tuning parameters:
  
#  mtry  Accuracy   Kappa    
#   7    0.777       0.55
#   8    0.778       0.56
#   9    0.777       0.55
#  10    0.777       0.55
#  11    0.777       0.55

# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 8.

eval_model_cv <- predict(rf_default_train_SMOTE, test, type = "prob")
perform_fun_rf_cv <- function(cutoff) 
{
  predicted_resp_cv <- as.factor(ifelse(eval_model_cv[, 2] >= cutoff, "1", "0"))
  conf <- confusionMatrix(predicted_resp_cv, test$Performance.Tag, positive = "0")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  OUT_rf_crv <- t(as.matrix(c(sens, spec, acc))) 
  colnames(OUT_rf_crv) <- c("sensitivity", "specificity", "accuracy")
  return(OUT_rf_crv)
}
  
summary(eval_model_cv[,2])
s3 = seq(.00,.91,length=100)
OUT_rf_crv = matrix(0,100,3)

# calculate the sens, spec and acc for different cutoff values
for(i in 1:100)
{
  OUT_rf_crv[i,] = perform_fun_rf_cv(s3[i])
} 

# plotting cutoffs

plot(s3, OUT_rf_crv[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s3,OUT_rf_crv[,2],col="darkgreen",lwd=2)
lines(s3,OUT_rf_crv[,3],col=4,lwd=2)

abline(v=0.27)
box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))
min(abs(OUT_rf_crv[,1]-OUT_rf_crv[,2]))
cutoff_rf_crv <- s3[which(abs(OUT_rf_crv[,1]-OUT_rf_crv[,2])<0.50)]
cutoff_rf_crv

#[1] 0.04595960 0.05515152 0.06434343 0.07353535 0.08272727 0.09191919 0.10111111 0.11030303 0.11949495 0.12868687 0.13787879
#[12] 0.14707071 0.15626263 0.16545455 0.17464646 0.18383838 0.19303030 0.20222222 0.21141414 0.22060606 0.22979798 0.23898990
#[23] 0.24818182 0.25737374 0.26656566 0.27575758 0.28494949 0.29414141 0.30333333 0.31252525 0.32171717 0.33090909 0.34010101
#[34] 0.34929293 0.35848485 0.36767677 0.37686869 0.38606061 0.39525253 0.40444444 0.41363636 0.42282828 0.43202020 0.44121212
#[45] 0.45040404 0.45959596 0.46878788 0.47797980 0.48717172 0.49636364 0.50555556 0.51474747 0.52393939 0.53313131 0.54232323
#[56] 0.55151515 0.56070707 0.56989899 0.57909091 0.58828283 0.59747475 0.60666667 0.61585859 0.62505051 0.63424242 0.64343434
#[67] 0.65262626 0.66181818 0.67101010 0.68020202 0.68939394 0.69858586 0.70777778 0.71696970 0.72616162 0.73535354

# Choosing a cutoff at 0.27

predicted_response_crv <- factor(ifelse(eval_model_cv[, 2] >= 0.27, "1", "0"))
conf_forest_crv <- confusionMatrix(predicted_response_crv, test$Performance.Tag, positive = "0")
conf_forest_crv

# Confusion Matrix and Statistics

#          Reference
#Prediction     0     1
#         0   13895   402
#         1    6164   907

# Accuracy : 0.69          
# Kappa : 0.13          
# Sensitivity : 0.69
# Specificity : 0.69
        

#*** Variable Importance in Random Forest ***

importance(train_rf)

# Top five significant variables (by decreasing in MeanDecreaseGini)
#                                                                    MeanDecreaseGini
#No.of.times.30.DPD.or.worse.in.last.12.months                          2712.0422
#No.of.times.60.DPD.or.worse.in.last.6.months                           1975.4832
#No.of.times.30.DPD.or.worse.in.last.6.months                           1703.7273
#Outstanding.Balance                                                    1549.1174
#No.of.times.90.DPD.or.worse.in.last.12.months                          1381.4182


#------------------------------------------------------------------------------
# ** 2. Logistic regression (Non-rejected + Missing performance tag records) **
#------------------------------------------------------------------------------

#.............. 2.a) Model Building ...............#

model_1 = glm(Performance.Tag ~ ., data = train_SMOTE, family = "binomial")
summary(model_1)

# Variable selection using stepwise AIC algorithm for removing insignificant variables 
model_2<- stepAIC(model_1, direction="both")
summary(model_2)
#AIC: 59108
vif(model_2)
#all vif values are less than or almost equal to 2

#removing woe.Marital.Status..at.the.time.of.application..binned  variable as it is insignificant and has less P value
model_3<-glm(formula = Performance.Tag ~ Age + Income + No.of.months.in.current.residence + 
               No.of.months.in.current.company + woe.Profession.binned + 
               woe.Type.of.residence.binned + woe.Education.binned + 
               No.of.times.60.DPD.or.worse.in.last.6.months + No.of.times.30.DPD.or.worse.in.last.6.months + 
               No.of.times.90.DPD.or.worse.in.last.12.months + No.of.times.60.DPD.or.worse.in.last.12.months + 
               No.of.times.30.DPD.or.worse.in.last.12.months + Avgas.CC.Utilization.in.last.12.months + 
               No.of.trades.opened.in.last.6.months + No.of.trades.opened.in.last.12.months + 
               No.of.PL.trades.opened.in.last.6.months + No.of.PL.trades.opened.in.last.12.months + 
               Presence.of.open.home.loan, 
             family = "binomial", data = train_SMOTE)
summary(model_3)
vif(model_3)
#all vif values are less than or equal to 2
#AIC: 59109

#removing No.of.trades.opened.in.last.12.months  variable as it is insignificant and has less P value
model_4<-glm(formula = Performance.Tag ~ Age + Income + No.of.months.in.current.residence + 
               No.of.months.in.current.company + woe.Profession.binned + 
               woe.Type.of.residence.binned + woe.Education.binned + 
               No.of.times.60.DPD.or.worse.in.last.6.months + No.of.times.30.DPD.or.worse.in.last.6.months + 
               No.of.times.90.DPD.or.worse.in.last.12.months + No.of.times.60.DPD.or.worse.in.last.12.months + 
               No.of.times.30.DPD.or.worse.in.last.12.months + Avgas.CC.Utilization.in.last.12.months + 
               No.of.trades.opened.in.last.6.months +
               No.of.PL.trades.opened.in.last.6.months + No.of.PL.trades.opened.in.last.12.months + 
               Presence.of.open.home.loan, 
             family = "binomial", data = train_SMOTE)
summary(model_4)
vif(model_4)
#all vif values are less than or equal to 2
#AIC: 59110


#removing woe.Type.of.residence.binned  variable as it is insignificant and has less P value
model_5<-glm(formula = Performance.Tag ~ Age + Income + No.of.months.in.current.residence + 
               No.of.months.in.current.company + woe.Profession.binned + 
               woe.Education.binned + 
               No.of.times.60.DPD.or.worse.in.last.6.months + No.of.times.30.DPD.or.worse.in.last.6.months + 
               No.of.times.90.DPD.or.worse.in.last.12.months + No.of.times.60.DPD.or.worse.in.last.12.months + 
               No.of.times.30.DPD.or.worse.in.last.12.months + Avgas.CC.Utilization.in.last.12.months + 
               No.of.trades.opened.in.last.6.months +
               No.of.PL.trades.opened.in.last.6.months + No.of.PL.trades.opened.in.last.12.months + 
               Presence.of.open.home.loan, 
             family = "binomial", data = train_SMOTE)
summary(model_5)
vif(model_5)
#all vif values are less than or equal to 2
#AIC: 59113


#removing Presence.of.open.home.loan  variable as it is insignificant 
model_6<-glm(formula = Performance.Tag ~ Age + Income + No.of.months.in.current.residence + 
               No.of.months.in.current.company + woe.Profession.binned + 
               woe.Education.binned + 
               No.of.times.60.DPD.or.worse.in.last.6.months + No.of.times.30.DPD.or.worse.in.last.6.months + 
               No.of.times.90.DPD.or.worse.in.last.12.months + No.of.times.60.DPD.or.worse.in.last.12.months + 
               No.of.times.30.DPD.or.worse.in.last.12.months + Avgas.CC.Utilization.in.last.12.months + 
               No.of.trades.opened.in.last.6.months +
               No.of.PL.trades.opened.in.last.6.months + No.of.PL.trades.opened.in.last.12.months, 
             family = "binomial", data = train_SMOTE)
summary(model_6)
vif(model_6)
#all vif values are less than or almost equal to 2
#AIC: 59120
# Remaining all variables are significant

final_log_model <- model_6

#.............. 2.b) Model Evaluation ...............#

### 2.b.1) Confusion matrix:

test_pred_log = predict(final_log_model, type = "response",newdata = test[,-6])
summary(test_pred_log)

pred <- prediction(test_pred_log,test$Performance.Tag)
eva_log<-performance(pred,"sens","spec")
evaA_log<-performance(pred,'acc')

plot(evaA_log)
sensitivity <- eva_log@y.values[[1]]
cutoff <- eva_log@alpha.values[[1]]
specificity<- eva_log@x.values[[1]]
accuracy<-evaA_log@y.values[[1]]

plot(cutoff,sensitivity,col="red")
lines(cutoff,specificity,col="green")
lines(cutoff,accuracy,col="blue")
legend("bottomright", legend=c("sensitivity","accuracy","specificity"),
       col=c("red","blue","green"), lty=1:2, cex=0.8)

abline(v =0.46)
matrix<-data.frame(cbind(sensitivity,specificity,accuracy,cutoff))

final_matrix<-matrix[which(matrix$cutoff>0.1&matrix$cutoff<1),]
matrix<-sapply(final_matrix,function(x) round(x,2))
matrix<-data.frame(matrix)
head(matrix[which(matrix$accuracy==matrix$sensitivity&matrix$sensitivity==matrix$specificity),])

#choosing a final cutoff value of 0.46
#sensitivity=70%
#specificity=70%
#accuracy=70%

### Lets find cofusion matrix for this model
test_cutoff <- factor(ifelse(test_pred_log >=0.46, "1", "0"))
test_actual <- factor(ifelse(test$Performance.Tag==1, "1", "0"))
conf_final <- confusionMatrix(test_cutoff, test_actual, positive = "0")
conf_final

#Prediction     0     1
#         0 14001   404
#         1  6058   905

#Accuracy : 0.6976 
#Sensitivity : 0.6980          
#Specificity : 0.6914  


### 2.b.2) KS Statistics:

perf<-performance(pred,'tpr','fpr')
ks<-max(perf@y.values[[1]]-perf@x.values[[1]])
ks
# ks statistic for this model is 0.40


### 2.b.2) Area under the curve:
plot(roc(test_pred_log,factor(test$Performance.Tag)))
auc(roc(test_pred_log,factor(test$Performance.Tag)))
# area under ROC CURVE is 0.759


### 2.b.c) Gain, lift and KS chart:

gain<-performance(pred,'tpr','rpp')
deciles<-performance(pred,'rpp')
plot(gain)
abline(h=0.75,v=0.4)

plotLift(test_pred_log,test$Performance.Tag)
abline(h=2.8,v=2)
#a lift of 2.8 times is achieved with the model within first 2 deciles compared to random model

#within first 4 deciles as per the model we are able to predict 75% of defaulters correctly
#ks chart
k_stat_matrix<-data.frame(10*(deciles@y.values[[1]]),(perf@y.values[[1]]-perf@x.values[[1]]))
colnames(k_stat_matrix)[1]<-"deciles"
colnames(k_stat_matrix)[2]<-"k_statistic"
k_stat_matrix$k_statistic<-round(k_stat_matrix$k_statistic,2)
plot(k_stat_matrix)
abline(h=0.4,v=2.5)
#ks statistic lies withinin first 3 deciles 

#################################################

#### Cross validation on other test data set:
set.seed(50)
split_indices <- sample.split(master$Performance.Tag, SplitRatio = 0.70)

testds <- master[!split_indices, ]
test_pred_log_2 = predict(final_log_model, type = "response", newdata = testds[,-6])

#Lets find cofusion matrix for this model
test_cutoff <- factor(ifelse(test_pred_log_2 >=0.46, "1", "0"))
test_actual <- factor(ifelse(testds$Performance.Tag==1, "1", "0"))
conf_final <- confusionMatrix(test_cutoff, test_actual, positive = "0")
conf_final
#sensitivity=69%
#specificity=69%
#accuracy=69%

#################################################

#### Cross validation on other test data set:
set.seed(25)
split_indices <- sample.split(master$Performance.Tag, SplitRatio = 0.70)

testds1 <- master[!split_indices, ]
test_pred_log_3 = predict(final_log_model, type = "response",  newdata = testds1[,-6])

#Lets find cofusion matrix for this model
test_cutoff <- factor(ifelse(test_pred_log_3 >=0.46, "1", "0"))
test_actual <- factor(ifelse(testds1$Performance.Tag==1, "1", "0"))
conf_final <- confusionMatrix(test_cutoff, test_actual, positive = "0")
conf_final
#sensitivity=69%
#specificity=68%
#accuracy=69%


#========== Conclusion ==========
# For merged data with performance tag missing records, Logistic regression model is performing 
# better compared to Random forest. Hence considering Logistic regression model as final model for
# for application scorecard.


############################################################################################
# Application Scorecard for merged data including performance tag missing records(predicted)
############################################################################################

#Application score card for odds of 10 to 1 is 400. Score increases by 20 points for doubling odds. 
#Computing the probabilites of being defaulted for entire population

population = predict(final_log_model, type = "response", newdata = master[,-6])

#computing odds for good. Since the probability computed is for rejection (bad cusotmers),  Odd(good) =  (1-P(bad))/P(bad)
Odds_for_good<-sapply(population,function(x) (1-x)/x)

#computing  ln(odd(good))
ln_Odds_for_good<-sapply(Odds_for_good,function(x)log(x))

#Using the following formula for computing application score card
#400 + slope * (ln(odd(good)) - ln(10)) where slope is 20/(ln(20)-ln(10))
slope<-20/(log(20)-log(10))

application_score_card<-sapply(ln_Odds_for_good,function(x) 400 + slope * (x - log(10)))
head(application_score_card)

#making dataframe with score card
score_card_df<-cbind(master,application_score_card)
summary(application_score_card)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 272.7   328.1   349.5   344.1   364.7   393.4 
#scores range from 272.7 to 393.4 for applicants.
#high scores indicate less risk for defaulting
#mean score for approved customers is 344.1

###CUTOFF SCORE FOR ACCEPTING OR REJECTING AN APPLICATION

#cutoff selected for logistic regression model was 0.46
CUTOFF_SCORE<- 400 + (slope * (log((1-0.46)/0.46) - log(10)))
CUTOFF_SCORE
#CUTOFF SCORE is 338.18

ggplot(score_card_df,aes(application_score_card))+geom_histogram()
#Histogram clearly shows that the application score of applicants are below 338  are few which meets our expectation.
boxplot(application_score_card)

#No.of applicants above score 338.18
length(which(application_score_card>338.18))
#47790

#No.of applicants below score 338.18
length(which(application_score_card<338.18))
#23434

table(master_df$Performance.Tag)
#0        1 
#66853  2946

######## Application score for rejected dataset #########################
###### and score comparision between rejected and approved applicants ############

no_tags_new <- as.data.frame(no_tags)

population = predict(final_log_model, type = "response", newdata = no_tags_new[,-6])
Odds_for_good<-sapply(population,function(x) (1-x)/x)
ln_Odds_for_good<-sapply(Odds_for_good,function(x)log(x))

slope<-20/(log(20)-log(10))
application_score_card<-sapply(ln_Odds_for_good,function(x) 400 + slope * (x - log(10)))
summary(application_score_card)
boxplot(application_score_card)

#mean score for rejected applicants with missing performance tag is 297.7
#mean score for approved customers is 344.1
#Thus mean score of approved customers is much higher than rejected customers
head(application_score_card)

score_card_df<-cbind(no_tags_new,application_score_card)
summary(application_score_card)
CUTOFF_SCORE<- 400 + (slope * (log((1-0.46)/0.46) - log(10)))
CUTOFF_SCORE
#CUTOFF SCORE is 338.18

#Calculating Revenue Loss for the Bank
no_tags_new$Revenue_Loss <- ifelse(score_card_df$application_score_card>=338,1,0)
no_tags_new$Revenue_Loss<- as.factor(no_tags_new$Revenue_Loss)
summary(no_tags_new$Revenue_Loss)

##Total number of rejected customers by bank : 1425
#Customer should be given Credit card based on credit score
# No : 1419
# Yes : 6
# Credit Score Cut off : 338.18

###Revenue Loss for bank###
#Let us assume bank makes Rs.5000 per year from 1 credit card customer.
# Bank refused 6 potential credit card customer,amounting to Rs.30,000 annual loss to the bank. 

(1419/1425)
# 99% of the rejected records would default which meets our expectation.

###############################################################
#************** Financial benefit of the model ***************
###############################################################

# Financial Benefit of a model will be in terms of either
# a. decreasing the rejection the non-defaulters
# b. increasing the rejection of defaulters


#### Net Profit when no model is considered

#(Considering an average loss of Rs.5000 when each non defaulters application is rejected)
#and an average loss of Rs.1,00,000 when each accepted applicant defaults
Profit_without_model<-(66853*5000)-(2946*100000)
  #39665000

##### Profit with model on the dataset without records with no performance tag using confusion matrix

test_pred_log_final = predict(final_log_model, type = "response", newdata = master_df[,-6])
test_cutoff <- factor(ifelse(test_pred_log_final >=0.46, "1", "0"))
test_actual <- factor(ifelse(master_df$Performance.Tag==1, "1", "0"))
conf_final <- confusionMatrix(test_cutoff, test_actual, positive = "0")
conf_final

  #          Reference
  #Prediction     0     1
  #     0       45873  1311
  #     1       20980  1635
  #Profit with model will be total profit due to each true positive and each true negative
  #minus loss from each false positive and each false negative prediction

Profit_with_model<-(45873*5000)+(1635*100000)-(20980*5000)-(1311*100000)
  #156865000

#Net financial gain due to using our model
Net_financial_gain<-Profit_with_model-Profit_without_model
  #Rs.117200000

Percentage_financial_gain<-Net_financial_gain*100/Profit_without_model
  #295.4746% financial gain


### REVENUE LOSS:
#Revenue loss occurs when good customers are identified as bad and their credit card application is rejected.

#No of candidates rejected by the model who didn't default - 20980
#Total No of candidates who didn't default - 45873+20980 = 66853
Revenue_loss <- (20980/(66853))*100
  # 31.38% of the non defaulting customers are rejected which resulted in revenue loss.


### CREDIT LOSS:
#The candidates who have been selected by the bank and have defaulted are responsible for the credit loss to the bank.

# Credit loss without model
prop.table(table(master_df$Performance.Tag)) 

  #0          1 
  #0.95779309 0.04220691 
# % of candidates approved and then defaulted = 4.2%

# Credit loss with model 
# % of candidates approved and then defaulted = (1311/69799) =  1.8%

# Credit Loss Saved	= 4.2-1.8 = 2.4%

##################################################################################

#--------------------------------------------------------------------------------#
# *************************  Conclusions  ***************************************
#--------------------------------------------------------------------------------#

#1. Logistic regression model performs better than other models hence Logistic regression model is considered as final model.
    #Final Cutoff = 0.46
    #sensitivity=70%
    #specificity=70%
    #Accuracy = 70% 
    #KS statistics = 0.40
    #KS statistic lies within first 3 deciles
    #AUC = 0.759

#2. Significant variables:
    #Age 
    #Income
    #No.of.months.in.current.residence 
    #No.of.months.in.current.company 
    #woe.Profession.binned
    #woe.Education.binned 
    #No.of.times.60.DPD.or.worse.in.last.6.months
    #No.of.times.30.DPD.or.worse.in.last.6.months
    #No.of.times.90.DPD.or.worse.in.last.12.months
    #No.of.times.60.DPD.or.worse.in.last.12.months 
    #No.of.times.30.DPD.or.worse.in.last.12.months
    #Avgas.CC.Utilization.in.last.12.months
    #No.of.trades.opened.in.last.6.months
    #No.of.PL.trades.opened.in.last.6.months
    #No.of.PL.trades.opened.in.last.12.months

#3 Application scorecard is built on logistic regression model.
    # Cutoff score = 338.18

#4 Financial assesment
    #Net_financial_gain = #Rs.117200000
    #Percentage_financial_gain = #295.4746%
    #Revenue_loss = 31.38% of the non defaulting records are rejected which resulted in revenue loss.
    #Credit loss saved = 2.4%

#########################################  END  ####################################################