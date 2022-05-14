import numpy as np
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import urllib.request
import pandas as pd
import sys
import csv

csv.field_size_limit(sys.maxsize)


def prepareDataframe():
    # https://wortschatz.uni-leipzig.de/de/download
    # Make dataframe from input file
    df = pd.read_csv("inputData/deu_mixed-typical_2011_10K-words.txt", sep="\t", engine="python", encoding="utf-8",
                     on_bad_lines="skip", header=None, quotechar=' ')
    # Remove first 100 wordIDs (special characters)
    df = df[(df[0] >= 100)]

    # df[1] = df[1].str.replace('[@#&$%+-/*\\\\]', '')
    df[1] = df[1].str.replace('[^A-Za-z0-9]', '')

    # Remove unnecessary columns
    df = df.drop(columns=[0, 2])
    # Make everything lowercase
    df[1] = df[1].str.lower()
    # Remove duplicates and empty cells
    df = df.drop_duplicates()
    df[1].replace('', np.nan, inplace=True)
    df.dropna(subset=[1], inplace=True)
    # Reset index because of after drop the index does not reset
    df = df.reset_index()
    df = df.drop(columns=["index"])

    df.to_csv("inputData/dataframe.csv", sep="\t", encoding="utf-8")

    return df


def querySignDict(word):
    # Select your transport with a defined url endpoint
    transport = AIOHTTPTransport(url="https://signdict.org/graphql-api/graphiql")

    # Create a GraphQL client using the defined transport
    client = Client(transport=transport, fetch_schema_from_transport=True)

    # Provide a GraphQL query
    # Paste word in query from the for-loop in main
    query = gql(
        """
        {
          search(word: "%s") {
            id
            text
            type
            currentVideo {
              videoUrl
              license
              copyright
              originalHref
              user {
                name
              }
            }
          }
        }
    """ % word
    )

    # Execute the query on the transport
    result = client.execute(query)

    # Extract from result the videoUrl from every word (but every word can contain several videos)
    counter = 0
    for i in range(0, len(result["search"])):
        currentVideo = result["search"][i]
        videoUrl = currentVideo["currentVideo"]["videoUrl"]
        downloadVideo(videoUrl, word + "_" + str(counter))
        counter += 1


def downloadVideo(url, fileName):
    # /etc/resolv.conf --> nameserver 8.8.8.8
    # Download the video
    urllib.request.urlretrieve(url, "outputVideo/" + fileName + ".mp4")


if __name__ == '__main__':

    df_words = prepareDataframe()
    print(df_words)

    # Iterate through all the words of the dataframe
    for i in range(0, len(df_words)):
        single_word = df_words.iloc[i][1]
        if single_word:
            print(single_word)
            querySignDict(single_word)
        else:
            print("! Empty String !")
