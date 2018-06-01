# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 08:51:22 2017

@author: Abdulhakeem
"""

#""" uncomment this section

import findspark
findspark.init()
import pyspark as spark
sc = spark.SparkContext()

#till here """

# Import time
import time
# Import Spark CSV (https://github.com/databricks/spark-csv)
from pyspark.sql import SQLContext, Row
# Import statistics
from pyspark.sql.functions import mean, split, explode, regexp_extract, lit,\
    sum, count, col, round as rnd, collect_set, collect_list, udf
from pyspark.sql.types import LongType, FloatType, ArrayType, StringType
# Import machine library components for recommendation
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import PCA

# Import numpy
import numpy as np
# Import ast
import ast
# Import Bar Chart Tool for visual comparison
import grouped_bars as bars

# Path to the data on Disk
data_path = 'movie-rec-data/'

# Create SQLContext out of the SparkContext
sqlContext = SQLContext(sc)

"""
Tag Class allows to get tags on movies by users
"""
class Tag:
    # initialize and get movies tags
    def __init__(self):
        tagsDF = sqlContext.read.load(data_path + 'tags.csv',
                                format='com.databricks.spark.csv',
                                header='true',
                                inferschema='true')
        # Drop the tag and timestamp columns and persist
        self.tags = tagsDF.drop('tag', 'timestamp').cache()


"""
Rating Class allows to get ratings on movies by users
"""
class Rating:
    # initialize and get movies ratings
    def __init__(self):
        ratingsDF = sqlContext.read.load(data_path + 'ratings.csv',
                                format='com.databricks.spark.csv',
                                header='true',
                                inferschema='true')

        # Drop the timestamp column and persist
        self.ratings = ratingsDF.drop('timestamp').cache()

    # Movie average rating
    def movieAverage(self, movieId):
        return self.ratings.filter(self.ratings.movieId == movieId).select(mean('rating')).first()[0]

    # Get top n rated
    def topRatings(self, n):
        topRated = self.ratings.groupBy('movieId').avg('rating')
        return topRated.sort(col('rating'), ascending=False).\
            select('movieId', rnd(topRated[1], 2).alias('avg(rating)')).limit(n)


"""
Movie Class allows to get movies and genres watched by users
"""
class Movie:
    # initialize and get movies
    def __init__(self):
        moviesDF = sqlContext.read.load(data_path + 'movies.csv',
                                format='com.databricks.spark.csv',
                                header='true',
                                inferschema='true')
        self.movies = moviesDF.cache()
        self.tagged = Tag()
        self.rated = Rating()

        # Get all movies watched by all users, distinct of (ratings union of tags)
        self.usersMovies = self.rated.ratings.drop('rating').\
            union(self.tagged.tags).distinct().cache()

    # Get movies watched by a user
    def watched(self, userId):
        # Get the movies IDs watched by user from tags and ratings
        moviesId = self.usersMovies.filter(col('userId') == userId)
        return moviesId.join(self.movies, 'movieId')


    # Get movies watched by a list of users
    def watchedBy(self, userIds):
        usersMovies = []
        if(type(userIds) is not list):
            userIds = [userIds]

        for userId in userIds:
            # Get userMovies
            userMovies = self.watched(userId)
            # Get user movies genres
            userGenres = self.listGenres(userMovies).select('genre').distinct()
            usersMovies.append((userId, userMovies.count(), userGenres.count()))

        return sqlContext.createDataFrame(usersMovies, ['user', 'movies', 'genres'])

    # Get the favourite genre of a user
    def favGenre(self, userId):
        # Get movies IDs user has watched
        userMovies = self.watched(userId)
        # Get the genres and the counts perecentage
        userGenres = self.listGenres(userMovies).groupBy('genre').count()
        # Normalize the counts by using percentage fraction
        totalGenresCount = userGenres.select(sum('count')).first()[0]
        userNormalizedGenres = userGenres.select('genre', rnd(col('count')*100/totalGenresCount, 2).alias('percentage')).\
            join(self.allGenres(), 'genre', 'outer').fillna(0)
        return userNormalizedGenres

    # Get the favourite genre of each of a list of users
    def userFavGenre(self, userIds):
        usersGenres = []
        if(type(userIds) is not list):
            userIds = [userIds]

        for userId in userIds:
            favGenres = self.favGenre(userId).sort('percentage', ascending=False).limit(1)
            usersGenres += favGenres.withColumn('userId', lit(userId)).\
                select('userId', 'genre', 'percentage').collect()

        return sqlContext.createDataFrame(usersGenres)

    # Compare the movie tastes of two users
    def usersTastes(self, userIds):
        usersGenres = []
        for userId in userIds:
            # Add userId column
            favGenres = self.favGenre(userId).sort('genre', ascending=True).\
                withColumn('userId', lit(userId))

            usersGenres += favGenres.select('userId', 'genre', 'percentage').collect()

        return sqlContext.createDataFrame(usersGenres)

    # Get average rating and number of users for movie of id
    def movieById(self, movieId):
        # Get average rating
        avgRating = round(self.rated.movieAverage(movieId), 2)

        # Get the number of watchers from both users tags and ratings
        watchers = self.usersMovies.filter(col('movieId') == movieId).count()

        response = Row('movie', 'average', 'watchers')
        return sqlContext.createDataFrame([response(movieId, avgRating, watchers)])

    # Get average rating and number of users for movie of title
    def movieByIdOrTitle(self, movieId):
        if type(movieId) is str:
            movieId = self.movies.filter(self.movies.title == movieId).select('movieId').first().movieId
#            print(movieId)
        return self.movieById(movieId)

    # Get movie by year
    def movieInYear(self, year):
        return self.movies.select('movieId', 'title', 'genres', regexp_extract('title', "\(\d{4}\)", 0).alias('year')).\
            filter('year == "({})"'.format(year))

    # Split Genres
    def listGenres(self, movies):
        return movies.select(explode(split('genres', "\|").alias('genres')).alias('genre'), 'movieId', 'title')

    # Get all genres
    def allGenres(self):
        return self.listGenres(self.movies).select('genre').distinct().sort('genre', ascending=True)

    # Get all movies of genre
    def ofGenre(self, genre):
        genreMovies = self.listGenres(self.movies)

        return genreMovies.filter(genreMovies.genre == genre)

    # Get all movies of each of genres
    def moviesOfGenre(self, genres):
        genresMovies = []
        if(type(genres) is not list):
            genres = [genres]
        #[Row(f1=1, f2='row1'), Row(f1=2, f2='row2'), Row(f1=3, f2='row3')]
        for genre in genres:
            genresMovies += self.ofGenre(genre).collect()

        return sqlContext.createDataFrame(genresMovies, ['movieId', 'title', 'genre'])


    # Get the top n movies with rating
    def topRated(self, n):
        return self.rated.topRatings(n).join(self.movies, 'movieId')

    # Get the top n movies watched
    def topWatched(self, n):
        taggedMovies = self.tagged.tags.groupBy('movieId').count()
        ratedMovies = self.rated.ratings.groupBy('movieId').count()
        topMovies = taggedMovies.union(ratedMovies).groupBy('movieId').sum('count').sort('sum(count)', ascending=False).limit(n)

#        topMovies = self.usersMovies.drop('userId').groupBy('movieId').count().sort('count', ascending=True).limit(n)

        return topMovies.join(self.movies, 'movieId')


"""
Recommender Class allows to Build the recommendation model using ALS on the training data
"""
class Recommender:
    # initialize and train the system
    def __init__(self):
        self.ratings = Rating()
        self.movies = Movie()

        (training, test) = self.ratings.ratings.randomSplit([0.8, 0.2])

        # Build the recommendation model using ALS on the training data
        self.als = ALS(maxIter=5, regParam=0.01, rank=5, userCol="userId", itemCol="movieId", ratingCol="rating")
        self.model = self.als.fit(training)

        # Evaluate the model by computing the RMSE on the test data
        self.predictions = self.model.transform(test)
        self.predictions.cache()
        # Remove NaN values from prediction (due to SPARK-14489)
        self.predictions = self.predictions.filter(self.predictions.prediction != float('nan'))

    # Get the RMSE
    def evaluate(self):
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(self.predictions)
        return rmse

    # Recommend movies for a user
    def forUser(self, userId):
        # Select ratings for movies not watched by the user
        self.notWatched = self.ratings.ratings.filter(col('userId') != userId).drop('userId').withColumn('userId', lit(userId))
        self.predictionsForUser = self.model.transform(self.notWatched)
        recommended = self.predictionsForUser.filter(col('prediction') != float('nan')).\
            select('movieId', 'prediction').distinct().join(self.movies.movies.drop('genres'), 'movieId').\
                filter(col('prediction') <= 5).sort('prediction', ascending=False)
        return recommended


"""
Cluster Class allows to group users according to movie tastes
"""
class Cluster:
    # initialize and get clusters
    def __init__(self, Movie):
        # Get all movies watched by all users, distinct of (ratings union of tags)
        self.usersMovies = Movie.usersMovies
        # Join with movies to get genres
        self.usersGenres = self.usersMovies.join(Movie.movies, 'movieId').\
            select('userId', explode(split('genres', "\|").alias('genres')).alias('genre'))

        # All the 20 genres
        self.genres_str = 'Crime|Romance|Thriller|Adventure|Drama|War|Documentary|Fantasy|Mystery|Musical|Animation|Film-Noir|(no genres listed)|IMAX|Horror|Western|Comedy|Children|Action|Sci-Fi'
        # Get all users
        self.users = Movie.usersMovies.select('userId').distinct()
        # Form a template with users X genres
        self.usersGenresTemplate = self.users.withColumn('genres', lit(self.genres_str)).\
            select('userId', explode(split('genres', "\|").alias('genres')).alias('genre'))

        # Fill in the template with the actual values, and zero where null
        self.usersGenresFilled = self.usersGenres.groupBy('userId', 'genre').agg(count('genre').alias('count')).\
            join(self.usersGenresTemplate, ['userId', 'genre'], 'right').fillna(0)

        # Sort by Genre and form genre array and counts
        self.usersFeatures = self.usersGenresFilled.groupBy('userId', 'genre').agg(sum('count').alias('count_')).\
            sort('genre', ascending=True).groupBy('userId').\
                agg(collect_list('genre').alias('genres'), collect_list('count_').alias('count')).cache()

        #
        userGenres = self.usersFeatures.drop('genres')
        self.datapoints = userGenres.select('userId', normalizeUdf(col('count')).alias('features'))
        # Trains a k-means model.
        kmeans = KMeans(maxIter=10).setK(3).setSeed(1)
        self.model = kmeans.fit(self.datapoints.select('features'))
#        kmeans.save(data_path + "/kmeans")

        # PCA reduction for visual
        pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
        self.pcaModel = pca.fit(self.datapoints.select('features'))
#        pca.save(data_path + "/pca")

    # Evaluate clustering by computing Within Set Sum of Squared Errors.
    def evaluate(self):
        self.wssse = self.model.computeCost(self.datapoints.select('features'))
        return self.wssse

    # Get centers
    def getCenters(self):
        centers = self.model.clusterCenters()
        return centers

    # Group users
    def groupUsers(self):
        self.groupedUsers = self.model.transform(self.datapoints)
        return self.groupedUsers

    # PCA transform
    def pcaTransform(self):
        result = self.pcaModel.transform(self.datapoints.select('features')).select("pcaFeatures")
        return result

    # PCA visualization
    def visualize(self):
        self.reducedPoints = self.pcaModel.transform(self.groupUsers())
        return self.reducedPoints


# Only load once, for optimization
if 'movies' not in globals():
    movies = Movie()


# My defined function for normalization
def normalize(g):
    frac = []
    total = 0
    for acc in g:
        total += acc

    for c in g:
        frac.append(round(c/total,2))

    return Vectors.dense(frac)
# Register the UDF
normalizeUdf = udf(normalize, VectorUDT())


#______________________________________________________________________________
#______________________________________________________________________________
# Menu
stop = False

# While not exit!
while not stop:
    print("\n\r______________________________________________________________________")
    print("#######################################################################")
    print("\n\rMovies Rating/Tagging System")
    print("______________________________________________________________________")
    print("#######################################################################")
    print("\n\r[Features]:")
    print()
    print("1. Search user by id")           #, show the number of movies/genre that he/she has watched
    print("2. Search movie by id/title")    #, show the average rating, the number of users that have watched the movie
    print("3. Search genre")                #, show all movies in that genre
    print("4. Search movies by year")
    print("5. List the top n movies with highest rating")       #, ordered by the rating
    print("6. List the top n movies with the highest number of watches") #, ordered by the number of watches
    print("7. Find the favourite genre of a user") # or group of users
    print("8. Compare the movie tastes of two users")   #. Consider and justify how you will compare and present the data.
    print("9. Clustering and Recommendation System")

    print("0. Exit")

    option = int(input('Choose a Feature (e.g "3" for "Search genre"): '))

    # Processing
    start = 0
    def startTime():
        global start
        print("[Processing...please wait]")
        start = time.time()

    if option == 0:
        print("Stopping now... Bye!")
        stop = True
        break

    elif option == 1 or option == 7:
        response = ast.literal_eval(input('Enter User ID or a list of User IDs (e.g 1 or [1, 2]): '))
        startTime()
        movies.watchedBy(response).show() if option == 1 else movies.userFavGenre(response).show()

    elif option == 2:
        response = input('Enter Movie ID or Movie title (e.g 7 or "Sabrina (1995)"): ')
        if response.isnumeric:
            response = ast.literal_eval(response)
        startTime()
        movies.movieByIdOrTitle(response).show()

    elif option == 3:
        response = input('Enter Genre or a list of Genres (e.g Drama or ["Drama", "Action"]): ')
        startTime()
        if response[0] == "[":
            response = ast.literal_eval(response)

        movies.moviesOfGenre(response).show(truncate=40)

    elif option == 4:
        response = int(input('Enter a Year: '))
        startTime()
        movies.movieInYear(response).show()

    elif option == 5:
        response = int(input('Enter a number for top rating: '))
        startTime()
        movies.topRated(response).show()

    elif option == 6:
        response =  int(input('Enter a number for top watched: '))
        startTime()
        movies.topWatched(response).show()

    elif option == 8:
        response = ast.literal_eval(input('Enter a list of User IDs (e.g [1, 2]): '))
        startTime()
        dpoints = movies.usersTastes(response).toPandas().values

        fig = bars.plt.figure()
        ax = fig.add_subplot(111)
        bars.barplot(ax, np.array(dpoints))
        bars.plt.show()

    elif option == 9:
        print("Clustering Recommendation System")
        print()
        print("1. Cluster users according to movie tastes")
        print("2. Visualize clusters")
        print("3. Recommend movies for user")

        option = int(input('Choose a Feature (e.g "1" for "Evaluate Model"): '))

        if option == 1 or option == 2:
            if 'cluster' not in globals():
                print("Training the system for the first time")
                cluster = Cluster(movies)

        if option == 1:
            startTime()
            # Shows the result
            print("Within Set Sum of Squared Errors = " + str(cluster.evaluate()))
            print("Cluster Centers: ")
            index = 0
            for center in cluster.getCenters():
                print("center:",index)
                index +=1
                print(center)

        elif option == 2:
            startTime()
            # Shows the result
            dps_panda = cluster.visualize().drop('features').toPandas().values
            dps = np.array(dps_panda)

            colors = ['blue', 'red', 'yellow']

            for i in range(len(colors)):
                x = []
                y = []
                for dp in dps:
                    if dp[1] == i:
                        x.append(dp[2].values[0])
                        y.append(dp[2].values[1])

                bars.plt.scatter(x, y, c=colors[i])

            bars.plt.legend(['cluster'.format(i) for i in range(len(colors))], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            bars.plt.xlabel('First PC')
            bars.plt.ylabel('Second PC')
            bars.plt.title('PCA Scatter Plot')
            bars.plt.show()


        elif option == 3:
            if 'recommender' not in globals():
                print("Training the system for the first time")
                recommender = Recommender()
            response = ast.literal_eval(input('Enter User ID (e.g 1): '))
            startTime()
            print("Root-mean-square error = " + str(recommender.evaluate()))
            recommender.forUser(response).show()

    else:
        print("\n\r[Feature not available!]")

    print('\n\r::Processed in', time.time()-start, 'seconds.')
