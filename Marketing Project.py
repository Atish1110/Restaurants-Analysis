#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Calling different Libraries in python to work on the Determine certain matrices 
## to identify the star restaurants and generate recommendations.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt,seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings


# In[2]:


resturant_data=pd.read_excel('data.xlsx')  ## uploading resturant data


# In[3]:


resturant_data.head()


# In[4]:


country_code=pd.read_excel('country-code.xlsx')     ## uploading countru code
country_code.head()


# In[5]:


working_data=pd.merge(resturant_data,country_code,on='Country Code',how='left')
working_data.head()


# In[6]:


working_data.shape ## finding the shape of new matrix


# In[7]:


working_data.dtypes


# ### All data types are observed to be in the correct order

# In[8]:


working_data.isnull().sum()


# In[9]:


working_data=working_data.dropna(how='any')
print(working_data.isnull().sum())


#  ## We found that there is 1 null value in resturant name and there are 9 null values in cuisines

# In[10]:


working_data.duplicated()


# In[11]:


working_data.duplicated().sum()  ## finding total numbers of duplicates in each column


# ## We see that there are no duplicate values in data

# In[12]:


working_data1=working_data.rename(columns={'Restaurant ID':'resturant_id','Restaurant Name':'restaurant_name','City':'city',
                                           'Country Code':'country_code','Average Cost for two':'average_cost2',
                                           'Has Table booking':'table_booking','Has Online delivery':'deliver_online',
                                           'Price range':'price_range',
                                           'Aggregate rating':'agg_rating','Rating text':'rating_text','Votes':'votes',
                                           'Country':'country' })

## modifying the name of rows and columns for ease of working


# In[13]:


working_data1.columns      ## checking the modification of columns


# In[14]:


working_data1.city.value_counts()    # finding the number of restaurants city wise


# In[15]:


value_counts=working_data1.city.value_counts()                   # graphical represntation of number of restaurants city wise
filtered_value=value_counts[value_counts>20]
plt.figure(figsize=(12,6))
plt.bar(filtered_value.index,filtered_value.values)
plt.xlabel('city')
plt.ylabel('Num of restaurants')
plt.title('city wise distribution of restaurants')
plt.show()


# ## We can see that "New Delhi" is the city with maximum number of restaurants having total number of 5473 Restaurants
# ## Many cities have only 1 restaurants like "Miller","Weirton","Potrero","Monroe"

# In[16]:


working_data1.country.value_counts()   # finding the number of resturant by country


# In[17]:


value_counts=working_data1.country.value_counts()  # graphical representation of number of resturants by country
plt.figure(figsize=(20,8))
plt.bar(value_counts.index,value_counts.values)
plt.xlabel('country')
plt.ylabel('Num of restaurants')
plt.title('country wise distribution of restaurants')
plt.xticks(rotation=90)
plt.show()


# ## We observe that India has maximum number of restaurants while Canada has least.

# In[18]:


working_data1.restaurant_name.value_counts() # finding the count for franchise


# In[19]:


value_counts=working_data1.restaurant_name.value_counts()      # graphical representation of franchise by count
filtered_value=value_counts[value_counts>20]
plt.figure(figsize=(20,10))
plt.bar(filtered_value.index,filtered_value.values)
plt.xlabel('restaurant name')
plt.ylabel('Num of franchise')
plt.title('Restaurants with more than 20 fanchise')
plt.xticks(rotation=90)
plt.show()


# ## It is observed that Cafe Cofee Day with 83 branches.Dominos Pizza is a close second with 79 branches

# In[20]:


has_booking=working_data1.table_booking.value_counts() # finding the count of restaurants which have table bookingsvs dont
has_booking


# In[21]:


booking_ratio=has_booking[1]/has_booking[0]                 # finding the  table booking ratio
print('Ratio of restaurants haning bookings',booking_ratio)


# In[22]:


plt.bar(['No table booking','Table Booking'],has_booking)           # graphical represtation of table booking data
plt.xlabel('Table Booking')
plt.ylabel ('Number of Restaurants')
plt.title('Chart for restaurants having table bookings vs not having')
plt.show()


# ## Its observed that only 1158 restaurants have table bookings having a ratio of 0.13

# In[23]:


delivery=working_data1.deliver_online.value_counts()    # finding count of restaurants providing online delivery vs dont


# In[24]:


delivery=working_data1.deliver_online.value_counts()         ## finding the percentage of restaurants giving online delivery
delivery_avilable=delivery[1]
Num_of_restaurants=working_data1.resturant_id.count()
print('Number of restaurants having online delivery',delivery_avilable)
print ('Total Number of restaurants',Num_of_restaurants)
print ('percentage of restaurants having online delivery',(delivery_avilable/Num_of_restaurants)*100)


# In[25]:


delivery=working_data1.deliver_online.value_counts()          # graph reprentation of online delivery vs dont
plt.bar(['No Online delivery','Online Delivery'],delivery)
plt.xlabel('Restaurants having online delivery vs those that do not')
plt.ylabel('Number of Restaurants')
plt.title('Graph for online delivery of restaurants')
plt.show()


#  ## It is observed that only 25% of restaurants have online delivery

# In[26]:


deliver_votes=working_data1.loc[working_data1['deliver_online']=='Yes','votes'].sum()    # finding difference in votes of online delivery vs dont
print('online delivery total votes',deliver_votes)
no_deliver_votes=working_data1.loc[working_data1['deliver_online']=='No','votes'].sum()
print('no delivery total votes',no_deliver_votes)
difference_votes=[no_deliver_votes-deliver_votes]
print('Difference in votes for delivery and non delivery',difference_votes)


# In[27]:


grouped_data = working_data1.groupby('deliver_online')    # Get the number of votes for each group
votes_with_delivery = grouped_data.get_group('Yes')['votes'].sum()
votes_without_delivery = grouped_data.get_group('No')['votes'].sum()

# Create a bar chart

plt.bar(['With Delivery', 'Without Delivery'], [votes_with_delivery, votes_without_delivery])
plt.xlabel('Online Delivery')
plt.ylabel('Number of Votes')
plt.title('Number of Votes with and without Online Delivery')
plt.show()


# In[28]:


top_cuisines=working_data1.Cuisines.value_counts()  # displaying top 10 cuisines
top_cuisines.head(10)


# In[29]:


top_cuisines=working_data.Cuisines.value_counts().head(10)
plt.bar(top_cuisines.index,top_cuisines.values)
plt.xlabel('Cousines')
plt.ylabel('count')
plt.title('Top 10 cuisines with count')
plt.xticks(rotation=90)
plt.show()


# In[30]:


cuisines_count = working_data1.Cuisines.value_counts().head(3)   # finding the top 3 leading cuisines
cuisines_count


# ## It is observed that North Indian cuisine is the most served cuisine across all restaurants

# In[31]:


grouped_data = working_data1.groupby('city')         # getting most served cuisines across restaurants for each city
for city, group in grouped_data:
    top_cuisine = group['Cuisines'].value_counts().idxmax()
    print(f"Most served cuisine in {city}: {top_cuisine}")


# In[32]:


## Writing a function to convert all currency into USD


def convert_to_usd(currency, average_cost): 
    conversion_rates = {                                  # defining the current currency conversion rates for 1 usd
        'Indonesian Rupiah(IDR)': 14390.50,
        'Indian Rupees(Rs.)': 74.13,
        'Botswana Pula(P)': 11.07,
        'Sri Lankan Rupee(LKR)': 200,
        'Rand(R)': 15.39,
        'Qatari Rial(QR)': 3.64,
        'Dollar($)': 1,
        'Emirati Diram(AED)': 3.67,
        'Brazilian Real(R$)': 5.20,
        'Turkish Lira(TL)': 13.16,
        'Pounds(£)': 0.72,
        'NewZealand($)': 1.42
    }

    if currency in conversion_rates:                        
        conversion_rate = conversion_rates[currency]
        converted_cost = average_cost / conversion_rate
        return converted_cost
    else:
        return None
    
    


# In[33]:


working_data1['converted_cost_usd'] = working_data1.apply(lambda row: convert_to_usd(row['Currency'], 
                                      row['average_cost2']), axis=1)  # creating a column for converted currency

working_data1['converted_cost_usd'].head()


# In[34]:


sns.boxplot(data=working_data1,x='converted_cost_usd').set(title='Cost distribution across various restaurants')
average_price=working_data1['converted_cost_usd'].mean()           # finding the average rating
print()
print('Average Price across restaurants',average_price)


# ## It is observed that most restaurants have an price of below U.S.D50 with the and the mean is U.S.D 10.67 there are some restaurants where charges are high ranging from U.S.D 250 to U.S.D 500 and above

# In[35]:


working_data1.columns  #getting names of columns once again for further analysis


# ## Now we will analyse that how the various columns effect the Aggregate Rating for this we seggregate certain columns
# #  like for latitude,longitude,locality,locality verbose,address we can analyse city vs ratings also currency is name of currency with values inaverage cost

# In[36]:


sns.boxplot(data=working_data1,x='agg_rating').set(title='Boxplot of ratings')    # boxplot for rating
print()
average_rating = working_data1['agg_rating'].mean()               # finding the average rating
print('Average rating:', average_rating)


# ## It is observed that most ratings are between 2.5 to 3.5 with their mean at 2.66

# In[37]:


# Finding relation between price and rating through graph
plot = sns.jointplot(x=working_data1.converted_cost_usd, y=working_data1.agg_rating, kind='scatter')
plot.set_axis_labels('Price (USD)', 'Rating')
plot.fig.suptitle("Rating vs Price", y=1.02)


# ## It is observed that most of the scatter is around around 0 of x axis so we can say that rating does not increase with price

# In[38]:


plot = sns.jointplot(x=working_data1.votes, y=working_data1.agg_rating, kind='scatter') # code to plot rating vs votes
plot.set_axis_labels('Votes', 'Rating')
plot.fig.suptitle("Rating vs votes", y=1.02)


# ## From the plot its observed that rating does not increase when vote count increases

# In[39]:


sns.catplot(data=working_data1, x='table_booking', y='agg_rating', kind='bar', ci=None)
plt.xlabel('Table Booking')
plt.ylabel('Aggregate Rating')
plt.title('Rating Distribution by Table Booking')
plt.show()


# ## From the bar graph its observed that restaurants which have table booking have higher ratings

# In[40]:


plt.figure(figsize=[30, 650])
boxplot = sns.boxplot(y='Cuisines', x='agg_rating', data=working_data1, palette='Set2')
boxplot.set(title='Rating vs City')
boxplot.set_yticklabels(boxplot.get_yticklabels(), fontsize=20)

plt.show()


# ## It is observed that restuarants which serve mexican and filipino food have the maximum mean rating while resturants which serve only North Indian or Chinese have maximum variations in ratings

# In[41]:


plt.figure(figsize=[30, 100])
boxplot = sns.boxplot(y='city', x='agg_rating', data=working_data1, palette='Set2')
boxplot.set(title='Rating vs City')
boxplot.set_yticklabels(boxplot.get_yticklabels(), fontsize=20)

plt.show()


# ## Its observed that Pasig city has the highest mean rating while new delhi has maximum variations in rating

# In[42]:


sns.catplot(data=working_data1,x='deliver_online',y='agg_rating', kind='bar', ci=None)
plt.xlabel('Online Delivery')
plt.ylabel('Aggregate Rating')
plt.title('Rating Distribution by Delivery')
plt.show()


# ## It is observed that restaurants which have online delivery are better rated

# In[43]:


## checking for best average ratings across top 100 restaurants average ratings
top_restaurants = working_data1.groupby('restaurant_name')['agg_rating'].mean().nlargest(100)
top_restaurants = top_restaurants.reset_index()

plt.figure(figsize=(50,100))
ax = sns.barplot(data=top_restaurants, x='agg_rating', y='restaurant_name', palette='Set2')
ax.set_title('Average Rating by Top Restaurants', fontsize=30)
ax.set_xlabel('Average Rating', fontsize=30)
ax.set_ylabel('Restaurant Name', fontsize=30)
ax.tick_params(axis='y', labelsize=40)  # Adjust font size of y-axis tick labels
plt.show()


# ## It is observed that Atlanta Highway seafood market has the best average rating with Bao as close second

# In[44]:


sns.catplot(data=working_data1,x='Rating color',y='agg_rating', kind='bar', ci=None)
plt.xlabel('Rating Color')
plt.ylabel('Aggregate Rating')
plt.title('Aggregate rating color distribution')


# ## It is observed that dark green colour code has highest aggregate rating and white has least

# In[45]:


sns.catplot(data=working_data1,x='rating_text',y='agg_rating', kind='bar', ci=None)
plt.xlabel('Rating Text')
plt.ylabel('Aggregate Rating')
plt.title('Aggregate rating Text distribution')
plt.show()


# ## It is observed that most ratings are excellent

# In[46]:


sns.catplot(data=working_data1,x='price_range',y='agg_rating', kind='bar', ci=None)
plt.xlabel('Price Range')
plt.ylabel('Aggregate Rating')
plt.title('Aggregate rating Price Range distribution')
plt.show()


# ## It is observed that ratings are highest where price range is for 4

# In[47]:


plt.figure(figsize=(30, 50))
boxplot = sns.boxplot(y='agg_rating', x='country_code', data=working_data1, palette='Set2')
boxplot.set_title('Rating vs Country Code', fontsize=30)
boxplot.set_xticks(range(len(working_data1['country_code'].unique())))  # Set y-axis tick positions
boxplot.set_xticklabels(boxplot.get_xticklabels(), fontsize=30)
boxplot.tick_params(axis='x', labelsize=30)  # Increase font size of x-axis labels
plt.show()


# ## It is observed that country code 162 Phillipines has the highest mean in aggregate ratings while country code 1 which is INdia has the least

# In[ ]:





# In[ ]:




