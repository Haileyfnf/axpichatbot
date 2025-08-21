import pandas as pd
from settings import create_postgres_engine_by_sqlalchemy


def insert_data_to_table_from_excel():
	# 엑셀 읽기
	df = pd.read_excel('./excel/chatbot_datalist.xlsx')
	print(df)
	
	# DB에 넣기
	table_name = 'axpi_hailey_dataset'
	schema = 'pi_study'
	engine = create_postgres_engine_by_sqlalchemy()
	df.to_sql(
		name=table_name,
		con=engine,
		if_exists='replace',
		index=False,
		schema=schema
	)


if __name__ == '__main__':
	insert_data_to_table_from_excel()
