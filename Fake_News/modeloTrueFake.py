import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import TFAutoModel
import tensorflow as tf

true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

true_news['label'] = 1
fake_news['label'] = 0

df = pd.concat([true_news, fake_news], axis=0) #Concateno ambos dataframes
df = df.sample(frac=1).reset_index(drop=True) #Mezclo los datos 

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)


tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")# Cargar el tokenizer de BETO 
X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, max_length=512, return_tensors="tf")
X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, max_length=512, return_tensors="tf")
# Tokenizacion de los textos

y_train_tensor = tf.convert_to_tensor(y_train.values)
y_test_tensor = tf.convert_to_tensor(y_test.values)

# Cargar el modelo de BETO
beto_model = TFAutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

#añadir capa de clasificación
input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name="attention_mask")
#lo pasamos por el modelo BETO
embeddings = beto_model(input_ids, attention_mask=attention_mask)[0]
output = tf.keras.layers.Dense(1, activation="sigmoid")(embeddings[:, 0, :])

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output) #Creamos el modelo
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

print("Proesando el modelo...")
model.fit( {"input_ids": X_train_tokens["input_ids"], "attention_mask": X_train_tokens["attention_mask"]},
    y_train_tensor,
    validation_data=({"input_ids": X_test_tokens["input_ids"], "attention_mask": X_test_tokens["attention_mask"]},
        y_test_tensor,
    ),
    epochs=3,
    batch_size=16
)


nueva_noticia = "Ex-Presidente de Honduras Juan orlando condenado a 45 anios de prision en Estados Unidos por trafico ilegal de drogas ." #notifica de ejemplo


nueva_noticia_tokens = tokenizer(
    nueva_noticia,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="tf"
)

# Realizar la predicción
prediccion = model.predict(
    {"input_ids": nueva_noticia_tokens["input_ids"], "attention_mask": nueva_noticia_tokens["attention_mask"]}
)

# Mostrar el resultado
credibilidad = prediccion[0][0]
if credibilidad > 0.5:
    print(f"Noticia REAL con un índice de credibilidad de {credibilidad:.2f}")
else:
    print(f"Noticia FALSA con un índice de credibilidad de {1 - credibilidad:.2f}")
