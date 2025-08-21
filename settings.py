import pandas as pd

import psycopg2
from sqlalchemy import create_engine

# 전역변수
ENGINE_POSTGRES_AWS = "postgresql+psycopg2://postgres:fnf##)^2020!@fnf-process.ch4iazthcd1k.ap-northeast-2.rds.amazonaws.com:35430/postgres"
ENGINE_POSTGRES_REDSHIFT = "postgresql+psycopg2://data_user:Duser2022!#@prd-dt-redshift.conhugwtudej.ap-northeast-2.redshift.amazonaws.com:5439/fnf"

"""
판다스 2.1.4 써야합니다!!!!
Replacing it with version 2.1.4 made the error go away.

"""





def create_redshift_engine():
	# 차차가 알려줌
	return psycopg2.connect(
		host="prd-dt-redshift.conhugwtudej.ap-northeast-2.redshift.amazonaws.com",
		port="5439",
		user="dashff_user",
		password="Dashff2022!#",
		dbname="fnf",
	)


def create_postgres_engine():
	# 차차가 알려줌
	return psycopg2.connect(
		host='f-dt-process-env.cuhyn2zzixjx.ap-northeast-2.rds.amazonaws.com',
		port='35430',
		user='postgres',
		password='fnf##)^2020!',
		dbname='postgres',
	)


def create_postgres_engine_by_sqlalchemy():
	# SQLalchemy로 테스트해봄
	
	user = 'pi_study'
	password = 'TyH$36mdZh'
	host = 'f-dt-process-env.cuhyn2zzixjx.ap-northeast-2.rds.amazonaws.com'
	dbname = 'postgres'
	port = '35430'
	
	return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}")
