.PHONY: setup data features eda train risk test app api all
setup:    ; pip install -r requirements-dev.txt
data:     ; python -m src.make_dataset
features: ; python -m src.features
eda:      ; python -m src.eda
train:    ; python -m src.train --trials 12
risk:     ; python -m src.risk
test:     ; python -m pytest tests -q
app:      ; streamlit run app/app.py
api:      ; uvicorn api.main:app --reload
all: data features eda train risk test
