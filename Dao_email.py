import json

import pymysql

with open('conf.json') as f:
    config = json.load(f)


def connection_sql():
    conn = pymysql.connect(host='localhost', user=config["SQL_ID"], passwd=config["SQL_PASSWORD"], db=config["DB"])
    curs = conn.cursor()

    return conn, curs


def ham_add(email):
    conn, curs = connection_sql()
    fm = email['From']
    to = email['To']
    date = email['Date']
    subject = email['Subject']
    content = email['Content']
    query = '''
    CREATE TABLE IF NOT EXISTS ham(
    id INT PRIMARY KEY AUTO_INCREMENT,
    `from` LONGTEXT,
    `to` LONGTEXT,
    date VARCHAR(255),
    subject LONGTEXT NOT NULL,
    content LONGTEXT)
    '''

    curs.execute(query)
    conn.commit()

    query = "INSERT INTO ham(`from`,`to`, date, subject, content) VALUES(%s,%s,%s,%s,%s)"

    curs.execute(query, (fm, to, date, subject, content))
    conn.commit()

    conn.close()


def spam_add(email):
    conn, curs = connection_sql()
    fm = email['From']
    to = email['To']
    date = email['Date']
    subject = email['Subject']
    content = email['Content']
    query = '''
    CREATE TABLE IF NOT EXISTS spam(
    id INT PRIMARY KEY AUTO_INCREMENT,
    `from` LONGTEXT,
    `to` LONGTEXT,
    date VARCHAR(255),
    subject LONGTEXT NOT NULL,
    content LONGTEXT)
    '''

    curs.execute(query)
    conn.commit()

    query = "INSERT INTO spam(`from`,`to`, date, subject, content) VALUES(%s,%s,%s,%s,%s)"

    curs.execute(query, (fm, to, date, subject, content))
    conn.commit()

    conn.close()


def ham_get():
    conn, curs = connection_sql()

    query = "SELECT subject FROM ham"

    curs.execute(query)
    conn.commit()

    result = curs.fetchall()

    label = [True] * len(result)

    result = zip(result, label)

    conn.close()

    return result


def spam_get():
    conn, curs = connection_sql()

    query = "SELECT subject FROM spam"

    curs.execute(query)
    conn.commit()

    result = curs.fetchall()

    label = [False] * len(result)

    result = zip(result, label)

    conn.close()

    return result
