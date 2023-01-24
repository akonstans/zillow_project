# Zillow Project

## Project Description
Zillow is a tech real estate online marketplace with multiple listings and price estimation for properties around the United States.
The intention behind this porject is to create a predictive model to aid in assessing home value. The data used for this project is the historical data from 2017 for Single Family housing.

## Project Goals
- To discover drivers of housing prices from the feature set
- To create a Regression model that will be able to predict the housing price of Single Family Homes 
- The feature set used to predict house prices is aquired from Zillow
- The end goal of the model is to achieve predictions within 5% of actual price
- Deliver a report of the findings to Zillow team

## Questions to Answer
- Is either bedroom count or bathroom count a good indicator of price?
- Is total lot size a good driver of price?
- Is FIPS(location) a driver of price? 
- What is the largest driver behind price for the houses?

## Initial Thoughts and Hypothesis
I believe that neither bedroom count nor bathroom count will be a good indicator of home value since it does not take into account factors like total size, location, and other important factors. Perhaps combining the two into a single feature would be worthwhile. If it turns out to not be a good indicator even when combined the features likely can be dropped without significant impact on the model. I believe the largest driver behind price is going to be either total square footage or federal county id.

## Planning
- Create and use a wrangle.py to wrangle the data into a desired format
- Explore the data using a combination of statistical tests and graphs to discover drivers behind property price
- Develop a model that improves upon the baseline predicitons
  - The MVP feature set will be limited to bathrooms, bedrooms, squarefootage, yearbuilt, and location
- Draw and record conclusions
- Compile a concise report that will identify the most important discoveries in a report notebook that can be presented in less than 5 min

## Data Dictionary

|Feature | Definition|
|-----------------|-----------|
| tax_value | The total tax assessed value of the parcel |
| bedrooms |  Number of bedrooms in home|
| bathrooms |  Number of bathrooms in the home (can be fractional)|
| square_footage |  Calculated total finished living area of the home |
| yearbuilt |  The Year the home was built |
| square_footage |  Area of the lot in square feet |
| fips | Federal Information Processing Standard code or simply a county code |

## Takeaways and Conclusion
- Square footage and county location were the greatest drivers of home price
- The polynomial regression model performed the best
- My initial hypothesis was correct, however there is room for improvment on the model given more time
- If I had more time I would like to examine further features and use a couple different models to see if I can get more accurate results
