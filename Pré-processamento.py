{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1FdxZAH6TWPH"
      },
      "source": [
        "#**Exercícios - Aula 1**"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 1) Dado o dataset de produtos [1], descubra e desenvolva:\n",
        "\n",
        "[1] - https://dados-ml-pln.s3-sa-east-1.amazonaws.com/produtos.csv"
      ],
      "metadata": {
        "id": "fXEgdkBma6Qq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "df = pd.read_csv(\n",
        "    \"https://dados-ml-pln.s3-sa-east-1.amazonaws.com/produtos.csv\",\n",
        "    delimiter=\";\",\n",
        "    encoding='utf-8' )"
      ],
      "metadata": {
        "id": "b3xBLIOPIyCj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### 1.1. Analisar o % de valores nulos"
      ],
      "metadata": {
        "id": "EFdExcLpyvJn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "percentagem_nulos = (df.isnull().mean() * 100).round(2)\n",
        "print(percentagem_nulos)"
      ],
      "metadata": {
        "id": "HE0C9l8fyoE6",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "77ce7e80-adf4-45ff-a33f-27a01d4c90de"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "nome          0.00\n",
            "descricao    28.53\n",
            "categoria     0.00\n",
            "dtype: float64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### 1.2. Remover as linhas com valores nulos"
      ],
      "metadata": {
        "id": "dUP31qMIzJF_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "df = df.dropna()"
      ],
      "metadata": {
        "id": "OVcrSplkzRex"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### 1.3. Distribuição das “categorias”"
      ],
      "metadata": {
        "id": "OscRtUY3zRRI"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "distribuicao_categorias = df['categoria'].value_counts()\n",
        "print(distribuicao_categorias)\n",
        "#valores mostrado abaixo são referêntes ás quantidades após a remoção das linhas com valores nulos (Exercício 1.2)"
      ],
      "metadata": {
        "id": "ZOfPrW97zRCQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e04313c7-3031-4b5a-bdcc-1b0e4c2e815b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "livro        838\n",
            "maquiagem    788\n",
            "brinquedo    668\n",
            "game         622\n",
            "Name: categoria, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#### 1.4 Mostrar as 10 palavras que mais ocorrem na descrição\n"
      ],
      "metadata": {
        "id": "L-CbfxHHypAN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "todas_descricoes = ' '.join(df['descricao'].dropna())\n",
        "palavras = todas_descricoes.split()\n",
        "contagem_palavras = pd.Series(palavras).value_counts().head(10)\n",
        "print(contagem_palavras)"
      ],
      "metadata": {
        "id": "NMp70mZubnoZ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "89a6817b-14aa-4ed4-b225-b300fb9e0157"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "de      21579\n",
            "e       14331\n",
            "o       10019\n",
            "a        9743\n",
            "do       7760\n",
            "para     7136\n",
            "que      6420\n",
            "-        6288\n",
            "em       5912\n",
            "com      5173\n",
            "dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Gz6d4AqXTWPL"
      },
      "source": [
        "## 2) utilizando o df acima carregado, faça:"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "df = pd.read_csv(\n",
        "    \"https://dados-ml-pln.s3-sa-east-1.amazonaws.com/produtos.csv\",\n",
        "    delimiter=\";\",\n",
        "    encoding='utf-8' )"
      ],
      "metadata": {
        "id": "ZPHVrmEcIe0-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "s32ceDVmTWPL"
      },
      "source": [
        "#### 2.1. Elimine linhas com valores nulos"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KatXy-7hTWPL"
      },
      "outputs": [],
      "source": [
        "df = df.dropna()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lvcsAfnBTWPL"
      },
      "source": [
        "#### 2.2. Adicione uma nova coluna chamada texto, formada pela composição das colunas nome e descrição"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3H0Y6M2eTWPM",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 363
        },
        "outputId": "0e93f3cd-5b36-448e-88bc-316f6d4b0ae5"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                                                nome  \\\n",
              "0                            O Hobbit - 7ª Ed. 2013    \n",
              "1                 Livro - It A Coisa - Stephen King    \n",
              "2   Box  As Crônicas De Gelo E Fogo  Pocket  5 Li...   \n",
              "3                                  Box Harry Potter    \n",
              "4                          Livro Origem - Dan Brown    \n",
              "5   Mais Escuro - Cinquenta Tons Mais Escuros Pel...   \n",
              "6                      O Silmarillion - 5ª Ed. 2011    \n",
              "7                                O Pequeno Principe    \n",
              "8   Ed & Lorraine Warren - Demonologistas  Arquiv...   \n",
              "9            Box - Franz Kafka 1883-1924 - 3 Livros    \n",
              "\n",
              "                                           descricao categoria  \\\n",
              "0  Produto NovoBilbo Bolseiro é um hobbit que lev...     livro   \n",
              "1  Produto NovoDurante as férias escolares de 195...     livro   \n",
              "2  Produto NovoTodo o reino de Westeros ao alcanc...     livro   \n",
              "3  Produto Novo e Físico  A série Harry Potter ch...     livro   \n",
              "4  Produto NovoDe Onde Viemos? Para Onde Vamos? R...     livro   \n",
              "5  Produto Novo e Físico  O relacionamento quente...     livro   \n",
              "6  Produto NovoO Silmarillion, relata acontecimen...     livro   \n",
              "7  O Pequeno Príncipe é um dos personagens mais f...     livro   \n",
              "8  Produto NovoEles enfrentaram os mistérios mais...     livro   \n",
              "9  Produto NovoEste box contém 3 livros de Franz ...     livro   \n",
              "\n",
              "                                               texto  \n",
              "0   O Hobbit - 7ª Ed. 2013  Produto NovoBilbo Bol...  \n",
              "1   Livro - It A Coisa - Stephen King  Produto No...  \n",
              "2   Box  As Crônicas De Gelo E Fogo  Pocket  5 Li...  \n",
              "3   Box Harry Potter  Produto Novo e Físico  A sé...  \n",
              "4   Livro Origem - Dan Brown  Produto NovoDe Onde...  \n",
              "5   Mais Escuro - Cinquenta Tons Mais Escuros Pel...  \n",
              "6   O Silmarillion - 5ª Ed. 2011  Produto NovoO S...  \n",
              "7   O Pequeno Principe  O Pequeno Príncipe é um d...  \n",
              "8   Ed & Lorraine Warren - Demonologistas  Arquiv...  \n",
              "9   Box - Franz Kafka 1883-1924 - 3 Livros  Produ...  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-facf3730-6fc5-4088-9b99-543db381c52c\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>nome</th>\n",
              "      <th>descricao</th>\n",
              "      <th>categoria</th>\n",
              "      <th>texto</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>O Hobbit - 7ª Ed. 2013</td>\n",
              "      <td>Produto NovoBilbo Bolseiro é um hobbit que lev...</td>\n",
              "      <td>livro</td>\n",
              "      <td>O Hobbit - 7ª Ed. 2013  Produto NovoBilbo Bol...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Livro - It A Coisa - Stephen King</td>\n",
              "      <td>Produto NovoDurante as férias escolares de 195...</td>\n",
              "      <td>livro</td>\n",
              "      <td>Livro - It A Coisa - Stephen King  Produto No...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Box  As Crônicas De Gelo E Fogo  Pocket  5 Li...</td>\n",
              "      <td>Produto NovoTodo o reino de Westeros ao alcanc...</td>\n",
              "      <td>livro</td>\n",
              "      <td>Box  As Crônicas De Gelo E Fogo  Pocket  5 Li...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Box Harry Potter</td>\n",
              "      <td>Produto Novo e Físico  A série Harry Potter ch...</td>\n",
              "      <td>livro</td>\n",
              "      <td>Box Harry Potter  Produto Novo e Físico  A sé...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Livro Origem - Dan Brown</td>\n",
              "      <td>Produto NovoDe Onde Viemos? Para Onde Vamos? R...</td>\n",
              "      <td>livro</td>\n",
              "      <td>Livro Origem - Dan Brown  Produto NovoDe Onde...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>Mais Escuro - Cinquenta Tons Mais Escuros Pel...</td>\n",
              "      <td>Produto Novo e Físico  O relacionamento quente...</td>\n",
              "      <td>livro</td>\n",
              "      <td>Mais Escuro - Cinquenta Tons Mais Escuros Pel...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>O Silmarillion - 5ª Ed. 2011</td>\n",
              "      <td>Produto NovoO Silmarillion, relata acontecimen...</td>\n",
              "      <td>livro</td>\n",
              "      <td>O Silmarillion - 5ª Ed. 2011  Produto NovoO S...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>O Pequeno Principe</td>\n",
              "      <td>O Pequeno Príncipe é um dos personagens mais f...</td>\n",
              "      <td>livro</td>\n",
              "      <td>O Pequeno Principe  O Pequeno Príncipe é um d...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>Ed &amp; Lorraine Warren - Demonologistas  Arquiv...</td>\n",
              "      <td>Produto NovoEles enfrentaram os mistérios mais...</td>\n",
              "      <td>livro</td>\n",
              "      <td>Ed &amp; Lorraine Warren - Demonologistas  Arquiv...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>Box - Franz Kafka 1883-1924 - 3 Livros</td>\n",
              "      <td>Produto NovoEste box contém 3 livros de Franz ...</td>\n",
              "      <td>livro</td>\n",
              "      <td>Box - Franz Kafka 1883-1924 - 3 Livros  Produ...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-facf3730-6fc5-4088-9b99-543db381c52c')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-facf3730-6fc5-4088-9b99-543db381c52c button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-facf3730-6fc5-4088-9b99-543db381c52c');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-3daaa5a7-e132-4917-b63a-d12dfd575379\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-3daaa5a7-e132-4917-b63a-d12dfd575379')\"\n",
              "            title=\"Suggest charts.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-3daaa5a7-e132-4917-b63a-d12dfd575379 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 26
        }
      ],
      "source": [
        "df['texto'] = df['nome'] + ' ' + df['descricao']\n",
        "df.head(10)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jX3VZnUVTWPM"
      },
      "source": [
        "#### 2.3. Quantos Unigramas existem antes e depois de remover stopwords (use a coluna texto)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DyrqTNfSTWPM",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "b12e9446-ed7d-4113-d61b-8d6b9dbee1ff"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Nº de nigramas antes de remover as stopwords: 646125\n",
            "Nº de  unigramas depois de remover as stopwords: 459394\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import nltk\n",
        "nltk.download('punkt')\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.tokenize import word_tokenize\n",
        "\n",
        "qtd_desc = ' '.join(df['texto'].dropna())\n",
        "words = word_tokenize(qtd_desc)\n",
        "\n",
        "nltk.download('stopwords')\n",
        "stop_words = set(stopwords.words('portuguese'))\n",
        "\n",
        "words_no_stopwords = [words for words in words if words.lower() not in stop_words]\n",
        "total_unigramas = len(words)\n",
        "\n",
        "total_unigramas_sem_stopwords = len(words_no_stopwords)\n",
        "\n",
        "print(f\"Nº de nigramas antes de remover as stopwords: {total_unigramas}\")\n",
        "print(f\"Nº de  unigramas depois de remover as stopwords: {total_unigramas_sem_stopwords}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "a87hK5PtTWPM"
      },
      "source": [
        "#### 2.4. Quantos Bigramas existem antes e depois de remover stopwords (use a coluna texto)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tarAbm3VTWPM",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "115ec2ea-c7ae-4912-b19e-82e473dc062a"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n",
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Nº de bigramas antes de remover as stopwords: 646124\n",
            "Nº de bigramas após a remoção das stopwords: 459393\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import nltk\n",
        "nltk.download('punkt')\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.tokenize import word_tokenize\n",
        "from nltk.util import bigrams\n",
        "\n",
        "qtd_desc = ' '.join(df['texto'].dropna())\n",
        "word = word_tokenize(qtd_desc)\n",
        "\n",
        "nltk.download('stopwords')\n",
        "stop_words = set(stopwords.words('portuguese'))\n",
        "\n",
        "words_no_stopwords = [word for word in word if word.lower() not in stop_words]\n",
        "bigramas = list(bigrams(word))\n",
        "bigramas_so_stopwords = list(bigrams(words_no_stopwords))\n",
        "\n",
        "total_bigramas = len(bigramas)\n",
        "total_bigramas_sem_stopwords = len(bigramas_so_stopwords)\n",
        "\n",
        "print(f\"Nº de bigramas antes de remover as stopwords: {total_bigramas}\")\n",
        "print(f\"Nº de bigramas após a remoção das stopwords: {total_bigramas_sem_stopwords}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "39tU4nHuTWPN"
      },
      "source": [
        "#### 2.5. Quantos Trigramas existem antes e depois de remover stopwords (use a coluna texto)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VtS19R-CTWPN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "be343f95-9dbc-4e09-b919-6c4e50498b1c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Nº de trigramas antes de remover as stopwords: 646123\n",
            "Nº de trigramas após a remoção das stopwords: 459392\n"
          ]
        }
      ],
      "source": [
        "\n",
        "import pandas as pd\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.util import ngrams\n",
        "from nltk.tokenize import word_tokenize\n",
        "\n",
        "qtd_desc = ' '.join(df['texto'].dropna())\n",
        "word = word_tokenize(qtd_desc)\n",
        "\n",
        "nltk.download('stopwords')\n",
        "stop_words = set(stopwords.words('portuguese'))\n",
        "\n",
        "word_no_stopwords = [word for word in word if word.lower() not in stop_words]\n",
        "trigramas = list(ngrams(word, 3))\n",
        "trigramas_sem_stopwords = list(ngrams(word_no_stopwords, 3))\n",
        "\n",
        "total_trigramas = len(trigramas)\n",
        "total_trigramas_sem_stopwords = len(trigramas_sem_stopwords)\n",
        "\n",
        "print(f\"Nº de trigramas antes de remover as stopwords: {total_trigramas}\")\n",
        "print(f\"Nº de trigramas após a remoção das stopwords: {total_trigramas_sem_stopwords}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HcTMZN5fTWPN"
      },
      "source": [
        "#### 2.6. Quantos unigramas existem na coluna texto após aplicar Stemmer (utilize rslp)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cl9_xQpZTWPN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c3aa1b25-e350-48b8-9e91-67e56fb2a452"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package rslp to /root/nltk_data...\n",
            "[nltk_data]   Package rslp is already up-to-date!\n",
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Total de unigramas após aplicar o Stemmer RSLP: 459394\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import nltk\n",
        "nltk.download('rslp')\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.tokenize import word_tokenize\n",
        "from nltk.stem import RSLPStemmer\n",
        "\n",
        "qtd_desc = ' '.join(df['texto'].dropna())\n",
        "word = word_tokenize(qtd_desc)\n",
        "\n",
        "nltk.download('stopwords')\n",
        "stop_words = set(stopwords.words('portuguese'))\n",
        "\n",
        "\n",
        "words_no_stopwords = [word for word in word if word.lower() not in stop_words]\n",
        "stemmer = RSLPStemmer()\n",
        "word_stemmed = [stemmer.stem(word) for word in words_no_stopwords]\n",
        "\n",
        "total_unigramas_stemmed = len(word_stemmed)\n",
        "print(f\"Nº de unigramas após aplicar o Stemmer RSLP: {total_unigramas_stemmed}\")\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3 (ipykernel)",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.9.12"
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}