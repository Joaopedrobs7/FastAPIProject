def classificar_email(texto, modelo):
    """
    Recebe um texto e um modelo SetFit treinado.
    Retorna a classificação e as probabilidades.
    """
    # Predição de rótulo
    pred = modelo.predict([texto])[0]

    # Predição de probabilidades
    prob = modelo.predict_proba([texto])[0]

    # Converter tensor para float
    prod_prob = float(prob[1])
    improd_prob = float(prob[0])

    # Definir rótulo
    label = "Produtivo" if pred == 1 else "Improdutivo"

    return {
        "texto": texto,
        "label": label,
        "prob_produtivo": round(prod_prob*100,2),
        "prob_improdutivo": round(improd_prob*100,2)
    }

# # Exemplo de uso:
# texto_teste = "Preciso que você envie o relatório financeiro até amanhã."
# resultado = classificar_email(texto_teste, modelo)
# print(resultado)

