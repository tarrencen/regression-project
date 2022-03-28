### Regression Project

### Objective: 
As a junior data scientist on Zillow's data, produce a machine learning regression model that will predict property tax assessed values of single family residences that had a transaction in 2017. Provide a report on the process(es) employed to produce the model and their outcomes. Give a succinct presentation (no longer than five minutes) of the model, its results and any other relevant findings discovered in the process of creating the model.

### Planning:
- Acquire the appropriate data from the Codeup SQL database.
- Clean the data (addressing null values and incorrect data types)
- Do preliminary univariate analysis and accompanying visualizations to survey the data's distributions
- Prepare the data (feature selection and engineering, splitting(train/validate/test))
- Do bivariate analysis on interactions between features, with hypothesis testing and visualizations as needed (on test split only)
- Create, fit, and evaluate regression models, adjusting feature selection and performing feature engineering as necessary
- Evaluate best performing models on validate split; select best model and evaluate on test split
- Make recommendations and report findings

### Data dictionary:

'bathroomcnt (baths)':	 Number of bathrooms in home including fractional bathrooms

'bedroomcnt (beds)':	 Number of bedrooms in home 

'roomcnt (total_rooms)':	 Total number of rooms in the principal residence

'numberofstories':	 Number of stories or levels the home has

'buildingqualitytypeid':	 Overall assessment of condition of the building from best (lowest) to worst (highest)

'calculatedfinishedsquarefeet':	 Calculated total finished living area of the home

'fireplaceflag':	 Is a fireplace present in this home

'poolcnt':	 Number of pools on the lot (if any)

'fips':	 Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details

'lotsizesquarefeet':	 Area of the lot in square feet

'structuretaxvaluedollarcnt':	The assessed value of the built structure on the parcel

'landtaxvaluedollarcnt':	The assessed value of the land area of the parcel

'taxvaluedollarcnt':	The total tax assessed value of the parcel

'taxamount':	The total property tax assessed for that assessment year

'yearbuilt':	 The Year the principal residence was built

'logerror': $log(Zestimate) - log(Sale Price)$


### Initial questions:
- What features in a single family residence are most important in determining its tax assessed value?
- How does the number of rooms in a single family residence affect its tax assessed value?
- How does a single family residence's square footage (living area and/or lot size) affect its tax assessed value?
- Is there a relationship between a single family residence's condition rating ('buildingqualitytypeid') and its tax assessed value?