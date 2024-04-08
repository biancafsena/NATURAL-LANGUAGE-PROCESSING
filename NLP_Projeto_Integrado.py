{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "cMjfL3bUMYBc"
      },
      "source": [
        "## **Integrantes:**\n",
        "\n",
        "**348231** - BIANCA FIRMINO FERREIRA DE SENA\n",
        "\n",
        "**348051** - CLAUDIO RODRIGUES DOS SANTOS\n",
        "\n",
        "**348010** - DANIEL TOLEZANI CASTELO\n",
        "\n",
        "**347903** - PHELIPE CUSTODIO LIMA"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "#OBSERVAÇÃO: Para reproduzir os prompts de IA generativa (OpenAI) é necessário trocar a chave da API."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "g3k_Y7YHl9sM"
      },
      "source": [
        "# **Parte 1 – Criar um modelo classificador de assuntos aplicando técnicas tradicionais de NLP, que consiga classificar através de um texto o assunto conforme disponível na base de dados[1] para treinamento e validação do modelo seu modelo.**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "0y0wTFi5VL_1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Collecting openai\n",
            "  Obtaining dependency information for openai from https://files.pythonhosted.org/packages/07/de/ef3534d9417f7c72c75036fae6c85d9071aebbce8aa3616d3e69b9f0ca4d/openai-1.6.0-py3-none-any.whl.metadata\n",
            "  Downloading openai-1.6.0-py3-none-any.whl.metadata (17 kB)\n",
            "Collecting anyio<5,>=3.5.0 (from openai)\n",
            "  Obtaining dependency information for anyio<5,>=3.5.0 from https://files.pythonhosted.org/packages/bf/cd/d6d9bb1dadf73e7af02d18225cbd2c93f8552e13130484f1c8dcfece292b/anyio-4.2.0-py3-none-any.whl.metadata\n",
            "  Downloading anyio-4.2.0-py3-none-any.whl.metadata (4.6 kB)\n",
            "Collecting distro<2,>=1.7.0 (from openai)\n",
            "  Downloading distro-1.8.0-py3-none-any.whl (20 kB)\n",
            "Collecting httpx<1,>=0.23.0 (from openai)\n",
            "  Obtaining dependency information for httpx<1,>=0.23.0 from https://files.pythonhosted.org/packages/39/9b/4937d841aee9c2c8102d9a4eeb800c7dad25386caabb4a1bf5010df81a57/httpx-0.26.0-py3-none-any.whl.metadata\n",
            "  Downloading httpx-0.26.0-py3-none-any.whl.metadata (7.6 kB)\n",
            "Collecting pydantic<3,>=1.9.0 (from openai)\n",
            "  Obtaining dependency information for pydantic<3,>=1.9.0 from https://files.pythonhosted.org/packages/0a/2b/64066de1c4cf3d4ed623beeb3bbf3f8d0cc26661f1e7d180ec5eb66b75a5/pydantic-2.5.2-py3-none-any.whl.metadata\n",
            "  Downloading pydantic-2.5.2-py3-none-any.whl.metadata (65 kB)\n",
            "     ---------------------------------------- 0.0/65.2 kB ? eta -:--:--\n",
            "     ---------------------------------------- 65.2/65.2 kB 1.8 MB/s eta 0:00:00\n",
            "Collecting sniffio (from openai)\n",
            "  Downloading sniffio-1.3.0-py3-none-any.whl (10 kB)\n",
            "Requirement already satisfied: tqdm>4 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (4.66.1)\n",
            "Collecting typing-extensions<5,>=4.7 (from openai)\n",
            "  Obtaining dependency information for typing-extensions<5,>=4.7 from https://files.pythonhosted.org/packages/b7/f4/6a90020cd2d93349b442bfcb657d0dc91eee65491600b2cb1d388bc98e6b/typing_extensions-4.9.0-py3-none-any.whl.metadata\n",
            "  Downloading typing_extensions-4.9.0-py3-none-any.whl.metadata (3.0 kB)\n",
            "Requirement already satisfied: idna>=2.8 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from anyio<5,>=3.5.0->openai) (3.4)\n",
            "Requirement already satisfied: certifi in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpx<1,>=0.23.0->openai) (2023.5.7)\n",
            "Collecting httpcore==1.* (from httpx<1,>=0.23.0->openai)\n",
            "  Obtaining dependency information for httpcore==1.* from https://files.pythonhosted.org/packages/56/ba/78b0a99c4da0ff8b0f59defa2f13ca4668189b134bd9840b6202a93d9a0f/httpcore-1.0.2-py3-none-any.whl.metadata\n",
            "  Downloading httpcore-1.0.2-py3-none-any.whl.metadata (20 kB)\n",
            "Collecting h11<0.15,>=0.13 (from httpcore==1.*->httpx<1,>=0.23.0->openai)\n",
            "  Downloading h11-0.14.0-py3-none-any.whl (58 kB)\n",
            "     ---------------------------------------- 0.0/58.3 kB ? eta -:--:--\n",
            "     ---------------------------------------- 58.3/58.3 kB ? eta 0:00:00\n",
            "Collecting annotated-types>=0.4.0 (from pydantic<3,>=1.9.0->openai)\n",
            "  Obtaining dependency information for annotated-types>=0.4.0 from https://files.pythonhosted.org/packages/28/78/d31230046e58c207284c6b2c4e8d96e6d3cb4e52354721b944d3e1ee4aa5/annotated_types-0.6.0-py3-none-any.whl.metadata\n",
            "  Downloading annotated_types-0.6.0-py3-none-any.whl.metadata (12 kB)\n",
            "Collecting pydantic-core==2.14.5 (from pydantic<3,>=1.9.0->openai)\n",
            "  Obtaining dependency information for pydantic-core==2.14.5 from https://files.pythonhosted.org/packages/04/a1/36cea283ded0641e8c374cdcacfdab035c102467ac5ec721b7527c8ac1cf/pydantic_core-2.14.5-cp311-none-win_amd64.whl.metadata\n",
            "  Downloading pydantic_core-2.14.5-cp311-none-win_amd64.whl.metadata (6.6 kB)\n",
            "Requirement already satisfied: colorama in c:\\users\\phelipe custodio\\appdata\\roaming\\python\\python311\\site-packages (from tqdm>4->openai) (0.4.6)\n",
            "Downloading openai-1.6.0-py3-none-any.whl (225 kB)\n",
            "   ---------------------------------------- 0.0/225.4 kB ? eta -:--:--\n",
            "   ---------------------------------------- 225.4/225.4 kB 6.9 MB/s eta 0:00:00\n",
            "Downloading anyio-4.2.0-py3-none-any.whl (85 kB)\n",
            "   ---------------------------------------- 0.0/85.5 kB ? eta -:--:--\n",
            "   ---------------------------------------- 85.5/85.5 kB ? eta 0:00:00\n",
            "Downloading httpx-0.26.0-py3-none-any.whl (75 kB)\n",
            "   ---------------------------------------- 0.0/75.9 kB ? eta -:--:--\n",
            "   ---------------------------------------- 75.9/75.9 kB 4.4 MB/s eta 0:00:00\n",
            "Downloading httpcore-1.0.2-py3-none-any.whl (76 kB)\n",
            "   ---------------------------------------- 0.0/76.9 kB ? eta -:--:--\n",
            "   ---------------------------------------- 76.9/76.9 kB 4.2 MB/s eta 0:00:00\n",
            "Downloading pydantic-2.5.2-py3-none-any.whl (381 kB)\n",
            "   ---------------------------------------- 0.0/381.9 kB ? eta -:--:--\n",
            "   --------------------------------------- 381.9/381.9 kB 12.0 MB/s eta 0:00:00\n",
            "Downloading pydantic_core-2.14.5-cp311-none-win_amd64.whl (1.9 MB)\n",
            "   ---------------------------------------- 0.0/1.9 MB ? eta -:--:--\n",
            "   ----------- ---------------------------- 0.6/1.9 MB 11.5 MB/s eta 0:00:01\n",
            "   ----------------- ---------------------- 0.8/1.9 MB 13.2 MB/s eta 0:00:01\n",
            "   ----------------- ---------------------- 0.8/1.9 MB 13.2 MB/s eta 0:00:01\n",
            "   ------------------- -------------------- 0.9/1.9 MB 4.8 MB/s eta 0:00:01\n",
            "   ------------------- -------------------- 0.9/1.9 MB 4.8 MB/s eta 0:00:01\n",
            "   ------------------- -------------------- 0.9/1.9 MB 4.8 MB/s eta 0:00:01\n",
            "   ------------------- -------------------- 0.9/1.9 MB 3.0 MB/s eta 0:00:01\n",
            "   -------------------- ------------------- 1.0/1.9 MB 2.7 MB/s eta 0:00:01\n",
            "   -------------------- ------------------- 1.0/1.9 MB 2.7 MB/s eta 0:00:01\n",
            "   --------------------- ------------------ 1.0/1.9 MB 2.2 MB/s eta 0:00:01\n",
            "   --------------------------------- ------ 1.6/1.9 MB 3.2 MB/s eta 0:00:01\n",
            "   ---------------------------------------- 1.9/1.9 MB 3.5 MB/s eta 0:00:00\n",
            "Downloading typing_extensions-4.9.0-py3-none-any.whl (32 kB)\n",
            "Downloading annotated_types-0.6.0-py3-none-any.whl (12 kB)\n",
            "Installing collected packages: typing-extensions, sniffio, h11, distro, annotated-types, pydantic-core, httpcore, anyio, pydantic, httpx, openai\n",
            "Successfully installed annotated-types-0.6.0 anyio-4.2.0 distro-1.8.0 h11-0.14.0 httpcore-1.0.2 httpx-0.26.0 openai-1.6.0 pydantic-2.5.2 pydantic-core-2.14.5 sniffio-1.3.0 typing-extensions-4.9.0\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "[notice] A new release of pip is available: 23.2.1 -> 23.3.2\n",
            "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
          ]
        }
      ],
      "source": [
        "!pip install openai"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "5RalKZqoVMNv"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: openai in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (1.6.0)\n",
            "Requirement already satisfied: anyio<5,>=3.5.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (4.2.0)\n",
            "Requirement already satisfied: distro<2,>=1.7.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (1.8.0)\n",
            "Requirement already satisfied: httpx<1,>=0.23.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (0.26.0)\n",
            "Requirement already satisfied: pydantic<3,>=1.9.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (2.5.2)\n",
            "Requirement already satisfied: sniffio in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (1.3.0)\n",
            "Requirement already satisfied: tqdm>4 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (4.66.1)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.7 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (4.9.0)\n",
            "Requirement already satisfied: idna>=2.8 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from anyio<5,>=3.5.0->openai) (3.4)\n",
            "Requirement already satisfied: certifi in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpx<1,>=0.23.0->openai) (2023.5.7)\n",
            "Requirement already satisfied: httpcore==1.* in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpx<1,>=0.23.0->openai) (1.0.2)\n",
            "Requirement already satisfied: h11<0.15,>=0.13 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)\n",
            "Requirement already satisfied: annotated-types>=0.4.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic<3,>=1.9.0->openai) (0.6.0)\n",
            "Requirement already satisfied: pydantic-core==2.14.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic<3,>=1.9.0->openai) (2.14.5)\n",
            "Requirement already satisfied: colorama in c:\\users\\phelipe custodio\\appdata\\roaming\\python\\python311\\site-packages (from tqdm>4->openai) (0.4.6)\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "[notice] A new release of pip is available: 23.2.1 -> 23.3.2\n",
            "[notice] To update, run: python.exe -m pip install --upgrade pip\n",
            "\u001b[91mError:\u001b[0m Windows is not supported yet in the migration CLI\n"
          ]
        }
      ],
      "source": [
        "!pip install openai --upgrade\n",
        "!openai migrate"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "Ve2FV1jgXYoR"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: pandas in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (2.0.2)\n",
            "Requirement already satisfied: scikit-learn in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (1.3.0)\n",
            "Requirement already satisfied: openai in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (1.6.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas) (2023.3)\n",
            "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas) (2023.3)\n",
            "Requirement already satisfied: numpy>=1.21.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas) (1.24.3)\n",
            "Requirement already satisfied: scipy>=1.5.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-learn) (1.11.2)\n",
            "Requirement already satisfied: joblib>=1.1.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-learn) (1.3.2)\n",
            "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-learn) (3.2.0)\n",
            "Requirement already satisfied: anyio<5,>=3.5.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (4.2.0)\n",
            "Requirement already satisfied: distro<2,>=1.7.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (1.8.0)\n",
            "Requirement already satisfied: httpx<1,>=0.23.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (0.26.0)\n",
            "Requirement already satisfied: pydantic<3,>=1.9.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (2.5.2)\n",
            "Requirement already satisfied: sniffio in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (1.3.0)\n",
            "Requirement already satisfied: tqdm>4 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (4.66.1)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.7 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai) (4.9.0)\n",
            "Requirement already satisfied: idna>=2.8 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from anyio<5,>=3.5.0->openai) (3.4)\n",
            "Requirement already satisfied: certifi in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpx<1,>=0.23.0->openai) (2023.5.7)\n",
            "Requirement already satisfied: httpcore==1.* in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpx<1,>=0.23.0->openai) (1.0.2)\n",
            "Requirement already satisfied: h11<0.15,>=0.13 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)\n",
            "Requirement already satisfied: annotated-types>=0.4.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic<3,>=1.9.0->openai) (0.6.0)\n",
            "Requirement already satisfied: pydantic-core==2.14.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic<3,>=1.9.0->openai) (2.14.5)\n",
            "Requirement already satisfied: six>=1.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
            "Requirement already satisfied: colorama in c:\\users\\phelipe custodio\\appdata\\roaming\\python\\python311\\site-packages (from tqdm>4->openai) (0.4.6)\n",
            "Note: you may need to restart the kernel to use updated packages.\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "[notice] A new release of pip is available: 23.2.1 -> 23.3.2\n",
            "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
          ]
        }
      ],
      "source": [
        "pip install pandas scikit-learn openai"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "xDqxpK9E5Kw9"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Collecting openai==0.28\n",
            "  Obtaining dependency information for openai==0.28 from https://files.pythonhosted.org/packages/ae/59/911d6e5f1d7514d79c527067643376cddcf4cb8d1728e599b3b03ab51c69/openai-0.28.0-py3-none-any.whl.metadata\n",
            "  Downloading openai-0.28.0-py3-none-any.whl.metadata (13 kB)\n",
            "Requirement already satisfied: requests>=2.20 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai==0.28) (2.31.0)\n",
            "Requirement already satisfied: tqdm in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai==0.28) (4.66.1)\n",
            "Collecting aiohttp (from openai==0.28)\n",
            "  Obtaining dependency information for aiohttp from https://files.pythonhosted.org/packages/84/7a/70ca0dcffcb261d1e71590d1c93863f8b59415a52f610f75ee3e570e003c/aiohttp-3.9.1-cp311-cp311-win_amd64.whl.metadata\n",
            "  Downloading aiohttp-3.9.1-cp311-cp311-win_amd64.whl.metadata (7.6 kB)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests>=2.20->openai==0.28) (3.1.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests>=2.20->openai==0.28) (3.4)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests>=2.20->openai==0.28) (2.0.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests>=2.20->openai==0.28) (2023.5.7)\n",
            "Requirement already satisfied: attrs>=17.3.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from aiohttp->openai==0.28) (23.1.0)\n",
            "Collecting multidict<7.0,>=4.5 (from aiohttp->openai==0.28)\n",
            "  Downloading multidict-6.0.4-cp311-cp311-win_amd64.whl (28 kB)\n",
            "Collecting yarl<2.0,>=1.0 (from aiohttp->openai==0.28)\n",
            "  Obtaining dependency information for yarl<2.0,>=1.0 from https://files.pythonhosted.org/packages/27/41/945ae9a80590e4fb0be166863c6e63d75e4b35789fa3a61ff1dbdcdc220f/yarl-1.9.4-cp311-cp311-win_amd64.whl.metadata\n",
            "  Downloading yarl-1.9.4-cp311-cp311-win_amd64.whl.metadata (32 kB)\n",
            "Collecting frozenlist>=1.1.1 (from aiohttp->openai==0.28)\n",
            "  Obtaining dependency information for frozenlist>=1.1.1 from https://files.pythonhosted.org/packages/b3/21/c5aaffac47fd305d69df46cfbf118768cdf049a92ee6b0b5cb029d449dcf/frozenlist-1.4.1-cp311-cp311-win_amd64.whl.metadata\n",
            "  Downloading frozenlist-1.4.1-cp311-cp311-win_amd64.whl.metadata (12 kB)\n",
            "Collecting aiosignal>=1.1.2 (from aiohttp->openai==0.28)\n",
            "  Downloading aiosignal-1.3.1-py3-none-any.whl (7.6 kB)\n",
            "Requirement already satisfied: colorama in c:\\users\\phelipe custodio\\appdata\\roaming\\python\\python311\\site-packages (from tqdm->openai==0.28) (0.4.6)\n",
            "Downloading openai-0.28.0-py3-none-any.whl (76 kB)\n",
            "   ---------------------------------------- 0.0/76.5 kB ? eta -:--:--\n",
            "   ---------------------------------------- 76.5/76.5 kB 4.4 MB/s eta 0:00:00\n",
            "Downloading aiohttp-3.9.1-cp311-cp311-win_amd64.whl (364 kB)\n",
            "   ---------------------------------------- 0.0/364.8 kB ? eta -:--:--\n",
            "   --------------------------------------  358.4/364.8 kB 11.2 MB/s eta 0:00:01\n",
            "   --------------------------------------- 364.8/364.8 kB 11.4 MB/s eta 0:00:00\n",
            "Downloading frozenlist-1.4.1-cp311-cp311-win_amd64.whl (50 kB)\n",
            "   ---------------------------------------- 0.0/50.5 kB ? eta -:--:--\n",
            "   ---------------------------------------- 50.5/50.5 kB ? eta 0:00:00\n",
            "Downloading yarl-1.9.4-cp311-cp311-win_amd64.whl (76 kB)\n",
            "   ---------------------------------------- 0.0/76.7 kB ? eta -:--:--\n",
            "   ---------------------------------------- 76.7/76.7 kB 4.4 MB/s eta 0:00:00\n",
            "Installing collected packages: multidict, frozenlist, yarl, aiosignal, aiohttp, openai\n",
            "  Attempting uninstall: openai\n",
            "    Found existing installation: openai 1.6.0\n",
            "    Uninstalling openai-1.6.0:\n",
            "      Successfully uninstalled openai-1.6.0\n",
            "Successfully installed aiohttp-3.9.1 aiosignal-1.3.1 frozenlist-1.4.1 multidict-6.0.4 openai-0.28.0 yarl-1.9.4\n",
            "Note: you may need to restart the kernel to use updated packages.\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "[notice] A new release of pip is available: 23.2.1 -> 23.3.2\n",
            "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
          ]
        }
      ],
      "source": [
        "pip install openai==0.28"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "KfhOBBkA5b11"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Collecting llmx==0.0.15a0\n",
            "  Obtaining dependency information for llmx==0.0.15a0 from https://files.pythonhosted.org/packages/ad/5d/5a948ab901c621739f1b94225d0f33d6907a29f7875ff58eff8000deba82/llmx-0.0.15a0-py3-none-any.whl.metadata\n",
            "  Downloading llmx-0.0.15a0-py3-none-any.whl.metadata (8.1 kB)\n",
            "Requirement already satisfied: pydantic in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from llmx==0.0.15a0) (2.5.2)\n",
            "Requirement already satisfied: openai in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from llmx==0.0.15a0) (0.28.0)\n",
            "Collecting tiktoken (from llmx==0.0.15a0)\n",
            "  Obtaining dependency information for tiktoken from https://files.pythonhosted.org/packages/f1/62/73629527ff413c8ce20189d29eb52a91d6d4571e3214ef6d5a2c00ca4081/tiktoken-0.5.2-cp311-cp311-win_amd64.whl.metadata\n",
            "  Downloading tiktoken-0.5.2-cp311-cp311-win_amd64.whl.metadata (6.8 kB)\n",
            "Collecting diskcache (from llmx==0.0.15a0)\n",
            "  Obtaining dependency information for diskcache from https://files.pythonhosted.org/packages/3f/27/4570e78fc0bf5ea0ca45eb1de3818a23787af9b390c0b0a0033a1b8236f9/diskcache-5.6.3-py3-none-any.whl.metadata\n",
            "  Downloading diskcache-5.6.3-py3-none-any.whl.metadata (20 kB)\n",
            "Collecting cohere (from llmx==0.0.15a0)\n",
            "  Obtaining dependency information for cohere from https://files.pythonhosted.org/packages/ca/ca/92b7d1ec6cc79666f4275abdc8aefa5453b63c2a0d4e8383d10c05e81b6f/cohere-4.39-py3-none-any.whl.metadata\n",
            "  Downloading cohere-4.39-py3-none-any.whl.metadata (5.4 kB)\n",
            "Collecting google.auth (from llmx==0.0.15a0)\n",
            "  Obtaining dependency information for google.auth from https://files.pythonhosted.org/packages/f4/d2/9f6f3b9c0fd486617816cff42e856afea079d0bad99f0e60dc186c76b881/google_auth-2.25.2-py2.py3-none-any.whl.metadata\n",
            "  Downloading google_auth-2.25.2-py2.py3-none-any.whl.metadata (4.7 kB)\n",
            "Collecting typer (from llmx==0.0.15a0)\n",
            "  Downloading typer-0.9.0-py3-none-any.whl (45 kB)\n",
            "     ---------------------------------------- 0.0/45.9 kB ? eta -:--:--\n",
            "     ---------------------------------------- 45.9/45.9 kB 2.2 MB/s eta 0:00:00\n",
            "Requirement already satisfied: pyyaml in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from llmx==0.0.15a0) (6.0.1)\n",
            "Requirement already satisfied: aiohttp<4.0,>=3.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from cohere->llmx==0.0.15a0) (3.9.1)\n",
            "Collecting backoff<3.0,>=2.0 (from cohere->llmx==0.0.15a0)\n",
            "  Downloading backoff-2.2.1-py3-none-any.whl (15 kB)\n",
            "Collecting fastavro<2.0,>=1.8 (from cohere->llmx==0.0.15a0)\n",
            "  Obtaining dependency information for fastavro<2.0,>=1.8 from https://files.pythonhosted.org/packages/8c/73/a68d7460a8fbc0d05773cf33a20be902f755b471d7d1716f15e1cc756957/fastavro-1.9.1-cp311-cp311-win_amd64.whl.metadata\n",
            "  Downloading fastavro-1.9.1-cp311-cp311-win_amd64.whl.metadata (5.7 kB)\n",
            "Requirement already satisfied: importlib_metadata<7.0,>=6.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from cohere->llmx==0.0.15a0) (6.8.0)\n",
            "Requirement already satisfied: requests<3.0.0,>=2.25.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from cohere->llmx==0.0.15a0) (2.31.0)\n",
            "Requirement already satisfied: urllib3<3,>=1.26 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from cohere->llmx==0.0.15a0) (2.0.3)\n",
            "Collecting cachetools<6.0,>=2.0.0 (from google.auth->llmx==0.0.15a0)\n",
            "  Obtaining dependency information for cachetools<6.0,>=2.0.0 from https://files.pythonhosted.org/packages/a2/91/2d843adb9fbd911e0da45fbf6f18ca89d07a087c3daa23e955584f90ebf4/cachetools-5.3.2-py3-none-any.whl.metadata\n",
            "  Downloading cachetools-5.3.2-py3-none-any.whl.metadata (5.2 kB)\n",
            "Collecting pyasn1-modules>=0.2.1 (from google.auth->llmx==0.0.15a0)\n",
            "  Downloading pyasn1_modules-0.3.0-py2.py3-none-any.whl (181 kB)\n",
            "     ---------------------------------------- 0.0/181.3 kB ? eta -:--:--\n",
            "     -------------------------------------- 181.3/181.3 kB 5.5 MB/s eta 0:00:00\n",
            "Collecting rsa<5,>=3.1.4 (from google.auth->llmx==0.0.15a0)\n",
            "  Downloading rsa-4.9-py3-none-any.whl (34 kB)\n",
            "Requirement already satisfied: tqdm in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from openai->llmx==0.0.15a0) (4.66.1)\n",
            "Requirement already satisfied: annotated-types>=0.4.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic->llmx==0.0.15a0) (0.6.0)\n",
            "Requirement already satisfied: pydantic-core==2.14.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic->llmx==0.0.15a0) (2.14.5)\n",
            "Requirement already satisfied: typing-extensions>=4.6.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pydantic->llmx==0.0.15a0) (4.9.0)\n",
            "Requirement already satisfied: regex>=2022.1.18 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from tiktoken->llmx==0.0.15a0) (2023.10.3)\n",
            "Requirement already satisfied: click<9.0.0,>=7.1.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from typer->llmx==0.0.15a0) (8.1.7)\n",
            "Requirement already satisfied: attrs>=17.3.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from aiohttp<4.0,>=3.0->cohere->llmx==0.0.15a0) (23.1.0)\n",
            "Requirement already satisfied: multidict<7.0,>=4.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from aiohttp<4.0,>=3.0->cohere->llmx==0.0.15a0) (6.0.4)\n",
            "Requirement already satisfied: yarl<2.0,>=1.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from aiohttp<4.0,>=3.0->cohere->llmx==0.0.15a0) (1.9.4)\n",
            "Requirement already satisfied: frozenlist>=1.1.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from aiohttp<4.0,>=3.0->cohere->llmx==0.0.15a0) (1.4.1)\n",
            "Requirement already satisfied: aiosignal>=1.1.2 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from aiohttp<4.0,>=3.0->cohere->llmx==0.0.15a0) (1.3.1)\n",
            "Requirement already satisfied: colorama in c:\\users\\phelipe custodio\\appdata\\roaming\\python\\python311\\site-packages (from click<9.0.0,>=7.1.1->typer->llmx==0.0.15a0) (0.4.6)\n",
            "Requirement already satisfied: zipp>=0.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from importlib_metadata<7.0,>=6.0->cohere->llmx==0.0.15a0) (3.16.2)\n",
            "Collecting pyasn1<0.6.0,>=0.4.6 (from pyasn1-modules>=0.2.1->google.auth->llmx==0.0.15a0)\n",
            "  Obtaining dependency information for pyasn1<0.6.0,>=0.4.6 from https://files.pythonhosted.org/packages/d1/75/4686d2872bf2fc0b37917cbc8bbf0dd3a5cdb0990799be1b9cbf1e1eb733/pyasn1-0.5.1-py2.py3-none-any.whl.metadata\n",
            "  Downloading pyasn1-0.5.1-py2.py3-none-any.whl.metadata (8.6 kB)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3.0.0,>=2.25.0->cohere->llmx==0.0.15a0) (3.1.0)\n",
            "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3.0.0,>=2.25.0->cohere->llmx==0.0.15a0) (3.4)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from requests<3.0.0,>=2.25.0->cohere->llmx==0.0.15a0) (2023.5.7)\n",
            "Downloading llmx-0.0.15a0-py3-none-any.whl (19 kB)\n",
            "Downloading cohere-4.39-py3-none-any.whl (51 kB)\n",
            "   ---------------------------------------- 0.0/51.7 kB ? eta -:--:--\n",
            "   ---------------------------------------- 51.7/51.7 kB 2.8 MB/s eta 0:00:00\n",
            "Downloading diskcache-5.6.3-py3-none-any.whl (45 kB)\n",
            "   ---------------------------------------- 0.0/45.5 kB ? eta -:--:--\n",
            "   ---------------------------------------- 45.5/45.5 kB 1.1 MB/s eta 0:00:00\n",
            "Downloading google_auth-2.25.2-py2.py3-none-any.whl (184 kB)\n",
            "   ---------------------------------------- 0.0/184.2 kB ? eta -:--:--\n",
            "   --------------------------------------- 184.2/184.2 kB 10.9 MB/s eta 0:00:00\n",
            "Downloading tiktoken-0.5.2-cp311-cp311-win_amd64.whl (786 kB)\n",
            "   ---------------------------------------- 0.0/786.4 kB ? eta -:--:--\n",
            "   ---------------- ---------------------- 337.9/786.4 kB 10.6 MB/s eta 0:00:01\n",
            "   ------------------------------- -------- 614.4/786.4 kB 9.7 MB/s eta 0:00:01\n",
            "   ------------------------------- -------- 614.4/786.4 kB 9.7 MB/s eta 0:00:01\n",
            "   ------------------------------- -------- 614.4/786.4 kB 9.7 MB/s eta 0:00:01\n",
            "   ------------------------------- -------- 614.4/786.4 kB 9.7 MB/s eta 0:00:01\n",
            "   ------------------------------- -------- 614.4/786.4 kB 9.7 MB/s eta 0:00:01\n",
            "   -------------------------------- ------- 634.9/786.4 kB 2.2 MB/s eta 0:00:01\n",
            "   ---------------------------------- ----- 686.1/786.4 kB 2.0 MB/s eta 0:00:01\n",
            "   ---------------------------------------- 786.4/786.4 kB 2.1 MB/s eta 0:00:00\n",
            "Downloading cachetools-5.3.2-py3-none-any.whl (9.3 kB)\n",
            "Downloading fastavro-1.9.1-cp311-cp311-win_amd64.whl (498 kB)\n",
            "   ---------------------------------------- 0.0/498.4 kB ? eta -:--:--\n",
            "   ------------------- -------------------- 245.8/498.4 kB 7.6 MB/s eta 0:00:01\n",
            "   ---------------------------------------- 498.4/498.4 kB 7.7 MB/s eta 0:00:00\n",
            "Downloading pyasn1-0.5.1-py2.py3-none-any.whl (84 kB)\n",
            "   ---------------------------------------- 0.0/84.9 kB ? eta -:--:--\n",
            "   ---------------------------------------- 84.9/84.9 kB ? eta 0:00:00\n",
            "Installing collected packages: pyasn1, fastavro, diskcache, cachetools, backoff, typer, tiktoken, rsa, pyasn1-modules, google.auth, cohere, llmx\n",
            "Successfully installed backoff-2.2.1 cachetools-5.3.2 cohere-4.39 diskcache-5.6.3 fastavro-1.9.1 google.auth-2.25.2 llmx-0.0.15a0 pyasn1-0.5.1 pyasn1-modules-0.3.0 rsa-4.9 tiktoken-0.5.2 typer-0.9.0\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "[notice] A new release of pip is available: 23.2.1 -> 23.3.2\n",
            "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
          ]
        }
      ],
      "source": [
        "!pip install llmx==0.0.15a0"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "FEwKszvHcKLQ"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Collecting scikit-plotNote: you may need to restart the kernel to use updated packages.\n",
            "\n",
            "  Using cached scikit_plot-0.3.7-py3-none-any.whl (33 kB)\n",
            "Requirement already satisfied: matplotlib>=1.4.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-plot) (3.7.1)\n",
            "Requirement already satisfied: scikit-learn>=0.18 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-plot) (1.3.0)\n",
            "Requirement already satisfied: scipy>=0.9 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-plot) (1.11.2)\n",
            "Requirement already satisfied: joblib>=0.10 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-plot) (1.3.2)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib>=1.4.0->scikit-plot) (1.0.7)\n",
            "Requirement already satisfied: cycler>=0.10 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib>=1.4.0->scikit-plot) (0.11.0)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib>=1.4.0->scikit-plot) (4.39.4)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib>=1.4.0->scikit-plot) (1.4.4)\n",
            "Requirement already satisfied: numpy>=1.20 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib>=1.4.0->scikit-plot) (1.24.3)\n",
            "Requirement already satisfied: packaging>=20.0 in c:\\users\\phelipe custodio\\appdata\\roaming\\python\\python311\\site-packages (from matplotlib>=1.4.0->scikit-plot) (23.1)\n",
            "Requirement already satisfied: pillow>=6.2.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib>=1.4.0->scikit-plot) (9.5.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib>=1.4.0->scikit-plot) (3.0.9)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib>=1.4.0->scikit-plot) (2.8.2)\n",
            "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from scikit-learn>=0.18->scikit-plot) (3.2.0)\n",
            "Requirement already satisfied: six>=1.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from python-dateutil>=2.7->matplotlib>=1.4.0->scikit-plot) (1.16.0)\n",
            "Installing collected packages: scikit-plot\n",
            "Successfully installed scikit-plot-0.3.7\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "[notice] A new release of pip is available: 23.2.1 -> 23.3.2\n",
            "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
          ]
        }
      ],
      "source": [
        "pip install scikit-plot"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "HA3vI_fjrjNO"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: seaborn in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (0.12.2)\n",
            "Requirement already satisfied: numpy!=1.24.0,>=1.17 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from seaborn) (1.24.3)\n",
            "Requirement already satisfied: pandas>=0.25 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from seaborn) (2.0.2)\n",
            "Requirement already satisfied: matplotlib!=3.6.1,>=3.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from seaborn) (3.7.1)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.0.7)\n",
            "Requirement already satisfied: cycler>=0.10 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (0.11.0)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (4.39.4)\n",
            "Requirement already satisfied: kiwisolver>=1.0.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (1.4.4)\n",
            "Requirement already satisfied: packaging>=20.0 in c:\\users\\phelipe custodio\\appdata\\roaming\\python\\python311\\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (23.1)\n",
            "Requirement already satisfied: pillow>=6.2.0 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (9.5.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (3.0.9)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from matplotlib!=3.6.1,>=3.1->seaborn) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas>=0.25->seaborn) (2023.3)\n",
            "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from pandas>=0.25->seaborn) (2023.3)\n",
            "Requirement already satisfied: six>=1.5 in c:\\users\\phelipe custodio\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.1->seaborn) (1.16.0)\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "[notice] A new release of pip is available: 23.2.1 -> 23.3.2\n",
            "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
          ]
        }
      ],
      "source": [
        "!pip install seaborn"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NsRtKxVMMYUU"
      },
      "source": [
        "### **Bibliotecas**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "2EtU37a9MYr5"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import openai\n",
        "import seaborn as sns\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import scikitplot as skplt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from sklearn.metrics import roc_curve, auc\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.metrics import accuracy_score, classification_report"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AvQOZN22MbDF"
      },
      "source": [
        "Com objetivo de importar varias bibliotecas com modulos de dados."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QxANxSosMbim"
      },
      "source": [
        "### **Carregando dados**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "ep-_PXYFMG0V"
      },
      "outputs": [],
      "source": [
        "url_tradicional = \"https://raw.githubusercontent.com/thiagonogueira/datasets/main/tickets_reclamacoes_classificados_one_line.csv\"\n",
        "df_tradicional = pd.read_csv(url_tradicional, delimiter=';')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5_VZjlA-MSSB"
      },
      "source": [
        "Possui objetivo de carregar dados de um arquivo CSV hospedado na internet para um DataFrame usando a biblioteca pandas em Python."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "h0zk5LkiZycm"
      },
      "source": [
        "### **Verificação de nomes das colunas**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lpNAeY5-ZymX",
        "outputId": "18043392-b5c3-4dc7-9429-ef2b9e39f5b8"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Nomes das Colunas: Index(['id_reclamacao', 'data_abertura', 'categoria', 'descricao_reclamacao'], dtype='object')\n"
          ]
        }
      ],
      "source": [
        "print(\"Nomes das Colunas:\", df_tradicional.columns)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4no-dEleaCK9"
      },
      "source": [
        " Possui a função de imprimir os nomes das colunas do DataFrame df_tradicional."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yIY8gjXKMqfe"
      },
      "source": [
        "### **Definição de coluna de texto**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "h3LNBg-TMquA"
      },
      "outputs": [],
      "source": [
        "text_column_tradicional = df_tradicional.columns[3]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CJITkyGiZOWN"
      },
      "source": [
        " Com o objetivo de atribuir o nome da terceira coluna do DataFrame df_tradicional à variável text_column_tradicional."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ofZ9acSpZOY9"
      },
      "source": [
        "### **Divisão de dados em treino e teste**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "tmpSJZXNZn1K"
      },
      "outputs": [],
      "source": [
        "train_data_tradicional, test_data_tradicional, train_labels_tradicional, test_labels_tradicional = train_test_split(\n",
        "    df_tradicional[text_column_tradicional],\n",
        "    df_tradicional['categoria'],\n",
        "    test_size=0.2,\n",
        "    random_state=42\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OvrLXQ6raOx5"
      },
      "source": [
        "Para dividir os dados do DataFrame df_tradicional em conjuntos de treinamento e teste, incluindo os textos das reclamações (text_column_tradicional) e as categorias (categoria)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gpVVvvbkaO1b"
      },
      "source": [
        "### **Criação de vetorizador TF-IDF**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "nYVIbOzyaSeB"
      },
      "outputs": [],
      "source": [
        "tfidf_vectorizer_tradicional = TfidfVectorizer(stop_words=None)\n",
        "train_vectors_tradicional = tfidf_vectorizer_tradicional.fit_transform(train_data_tradicional)\n",
        "test_vectors_tradicional = tfidf_vectorizer_tradicional.transform(test_data_tradicional)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JMWpubD2alHW"
      },
      "source": [
        "Realizar a vetorização dos textos de treinamento e teste usando a técnica TF-IDF."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xRa9zP8calXj"
      },
      "source": [
        "## **Treinamento do modelo**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 75
        },
        "id": "_efFm0Cpalyt",
        "outputId": "dbb6a8d0-2626-4609-d3c0-cf881dbd8c17"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MultinomialNB</label><div class=\"sk-toggleable__content\"><pre>MultinomialNB()</pre></div></div></div></div></div>"
            ],
            "text/plain": [
              "MultinomialNB()"
            ]
          },
          "execution_count": 7,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "clf_tradicional = MultinomialNB()\n",
        "clf_tradicional.fit(train_vectors_tradicional, train_labels_tradicional)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kacdIKrgazzV"
      },
      "source": [
        "Possui a função de criar e treinar um classificador Naive Bayes Multinomial usando os vetores de treinamento gerados pela técnica TF-IDF."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PQQwdAYDaz5h"
      },
      "source": [
        "### **Avaliação do modelo**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ATSxDGZva0Ea",
        "outputId": "c23df683-812e-4513-ae71-816912daebb3"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Acurácia Tradicional: 0.7240806642941874\n",
            "\n",
            "Relatório de Classificação Tradicional:\n",
            "                                      precision    recall  f1-score   support\n",
            "\n",
            "Cartão de crédito / Cartão pré-pago       0.74      0.76      0.75      1055\n",
            "            Hipotecas / Empréstimos       0.85      0.75      0.80       718\n",
            "                             Outros       1.00      0.02      0.04       437\n",
            "       Roubo / Relatório de disputa       0.72      0.78      0.75       958\n",
            "         Serviços de conta bancária       0.65      0.91      0.76      1047\n",
            "\n",
            "                           accuracy                           0.72      4215\n",
            "                          macro avg       0.79      0.64      0.62      4215\n",
            "                       weighted avg       0.76      0.72      0.69      4215\n",
            "\n"
          ]
        }
      ],
      "source": [
        "predictions_tradicional = clf_tradicional.predict(test_vectors_tradicional)\n",
        "print(\"Acurácia Tradicional:\", accuracy_score(test_labels_tradicional, predictions_tradicional))\n",
        "print(\"\\nRelatório de Classificação Tradicional:\\n\", classification_report(test_labels_tradicional, predictions_tradicional))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oZUxeAitPEBC"
      },
      "source": [
        "Realizar previsões usando o modelo treinado clf_tradicional nos dados de teste e, em seguida, imprimir métricas de avaliação do desempenho do modelo."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SVHvCklBb0vb"
      },
      "source": [
        "### **Calculando a Matriz de Confusão**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "uLiNgnt7eX23"
      },
      "outputs": [],
      "source": [
        "conf_matrix = confusion_matrix(test_labels_tradicional, predictions_tradicional)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zRfSh8S1k_jy"
      },
      "source": [
        "Com a função de calcular a matriz de confusão com base nas previsões do modelo clf_tradicional nos dados de teste."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TwJ_kTv7b1eI"
      },
      "source": [
        "### **Visualização - Matriz de Confusão**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "1WXfQGMUb2HD"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA3MAAAMNCAYAAAAyYS3LAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAADpNklEQVR4nOzddVgVaRsG8PvQ3UgoCgoqKIItYq642K1rg13YYneAhSIG5iLuGuvaa6NrIjY2dqBSCgLSdb4/1KOzgKKfeBi5f15zXZ6Zd2aemaMHnvO87zsSqVQqBREREREREYmKgrwDICIiIiIioq/HZI6IiIiIiEiEmMwRERERERGJEJM5IiIiIiIiEWIyR0REREREJEJM5oiIiIiIiESIyRwREREREZEIMZkjIiIiIiISISZzREREREREIsRkjoiISGRmzpwJiURSqOeQSCSYOXNmoZ7jR1u0aBHKli0LRUVFODo6Fso5xo0bB21tbbi5uSEuLg52dna4du1aoZyLiIjJHBERUT42btwIiUQCiUSCs2fP5toulUphYWEBiUSCVq1afdM5vLy8sGfPnv8zUnHIzs5GQEAAGjVqBAMDA6iqqsLS0hJ9+vTB5cuXC/XcR48exfjx4+Hs7IyAgAB4eXl993MkJSXB398fs2fPxu3bt2FkZAQtLS1UqVLlu5+LiAhgMkdERPRFampq2LJlS671p06dwosXL6CqqvrNx/6WZG7q1KlITU395nPKQ2pqKlq1aoW+fftCKpVi8uTJ8Pf3R+/evRESEoJatWrhxYsXhXb+f//9FwoKCtiwYQN69+6NFi1afPdzqKmp4c6dOxg9ejQuX76MFy9e4Pz581BQ4K9bRFQ4lOQdABERUVHXokUL/P333/Dz84OS0scfnVu2bEH16tXx+vXrHxJHcnIyNDU1oaSkJIhDDDw9PXH48GEsXboUo0aNEmybMWMGli5dWqjnj4mJgbq6OlRUVArtHEpKSihTpozstbm5eaGdi4gIYGWOiIjoi7p164bY2FgEBQXJ1mVkZGDHjh3o3r17nvssXrwYdevWhaGhIdTV1VG9enXs2LFD0EYikSA5ORmBgYGy7pzu7u4APo6Lu3PnDrp37w59fX3Uq1dPsO0Dd3d32f7/Xb407i09PR2jR4+GsbExtLW10aZNm3wrZC9fvkTfvn1hYmICVVVVVKpUCb///vuXbh9evHiBNWvWoGnTprkSOQBQVFTEuHHjUKpUKdm60NBQNG/eHDo6OtDS0kKTJk1w/vx5wX4fusEGBwdjzJgxMDY2hqamJtq3b49Xr17J2kkkEgQEBCA5OVl2XzZu3IinT5/K/v5f/713b9++xahRo2BpaQlVVVWUKFECTZs2xdWrV2VtTp48iU6dOqF06dJQVVWFhYUFRo8enWcV9d9//0X9+vWhqakJPT09tG3bFmFhYV+8l0REnxLX13pERERyYGlpCScnJ2zduhXNmzcHABw6dAgJCQno2rUr/Pz8cu2zbNkytGnTBj169EBGRga2bduGzp07Y//+/WjZsiUA4I8//kD//v1Rq1YtDBw4EABQrlw5wXE6d+4MGxsbeHl5QSqV5hnfoEGD4OLiIlh3+PBhbN68GSVKlPjstfXv3x9//vknunfvjrp16+Lff/+Vxfep6Oho1KlTBxKJBB4eHjA2NsahQ4fQr18/JCYm5pmkfXDo0CFkZWWhV69en43lg9u3b6N+/frQ0dHB+PHjoaysjDVr1qBRo0Y4deoUateuLWg/fPhw6OvrY8aMGXj69Cl8fX3h4eGBv/76C8C7+7x27VpcvHgR69evBwDUrVu3QLF8MHjwYOzYsQMeHh6ws7NDbGwszp49i7CwMFSrVg0AsH37dqSmpmLo0KEwMDDAxYsXsXz5crx48QJ///237FjHjh1D8+bNUbZsWcycOROpqalYvnw5nJ2dcfXqVVhaWn5VbERUjEmJiIgoTwEBAVIA0kuXLklXrFgh1dbWlqakpEilUqm0c+fO0saNG0ulUqm0TJky0pYtWwr2/dDug4yMDGnlypWlv/zyi2C9pqam1M3NLde5Z8yYIQUg7datW77b8vPgwQOprq6utGnTptKsrKx82127dk0KQDp06FDB+u7du0sBSGfMmCFb169fP6mZmZn09evXgrZdu3aV6urq5rreT40ePVoKQBoaGppvm0+1a9dOqqKiIn306JFsXUREhFRbW1vaoEED2boP74+Li4s0JydHcD5FRUVpfHy8bJ2bm5tUU1NTcJ4nT55IAUgDAgJyxfDf69fV1ZUOGzbss3EnJyfnWuft7S2VSCTSZ8+eydY5OjpKS5QoIY2NjZWtu379ulRBQUHau3fvz56DiOhT7GZJRERUAF26dEFqair279+Pt2/fYv/+/fl2sQQAdXV12d/fvHmDhIQE1K9fX9AtryAGDx78Ve2Tk5PRvn176OvrY+vWrVBUVMy37cGDBwEAI0aMEKz/b5VNKpVi586daN26NaRSKV6/fi1bXF1dkZCQ8NnrSkxMBABoa2t/Mf7s7GwcPXoU7dq1Q9myZWXrzczM0L17d5w9e1Z2vA8GDhwo6HZav359ZGdn49mzZ188X0Hp6enhwoULiIiIyLeNhoaG7O/Jycl4/fo16tatC6lUitDQUABAZGQkrl27Bnd3dxgYGMjaV6lSBU2bNpW9J0REBcFulkRERAVgbGwMFxcXbNmyBSkpKcjOzkanTp3ybb9//37MnTsX165dQ3p6umz91z4fzsrK6qvaDxgwAI8ePcK5c+dgaGj42bbPnj2DgoJCrq6dFSpUELx+9eoV4uPjsXbtWqxduzbPY8XExOR7Hh0dHQDvxp19yatXr5CSkpIrBgCwtbVFTk4Onj9/jkqVKsnWly5dWtBOX18fwLsk+ntZuHAh3NzcYGFhgerVq6NFixbo3bu3IOEMDw/H9OnTsW/fvlznTkhIAABZgpnf9R05ckQ20Q0R0ZcwmSMiIiqg7t27Y8CAAYiKikLz5s2hp6eXZ7szZ86gTZs2aNCgAVatWgUzMzMoKysjICAgz0ccfM6nFb4vWbZsGbZu3Yo///zzuz4UOycnBwDQs2dPuLm55dnmc89Sq1ixIgDg5s2bhfKw7vyqj9J8xhh+kF9inZ2dnWtdly5dUL9+fezevRtHjx7FokWLsGDBAuzatQvNmzdHdnY2mjZtiri4OEyYMAEVK1aEpqYmXr58CXd3d9k9JCL6npjMERERFVD79u0xaNAgnD9/Xja5Rl527twJNTU1HDlyRPAMuoCAgFxtv7ZSl58zZ85g3LhxGDVqFHr06FGgfcqUKYOcnBw8evRIUCm6d++eoN2HmS6zs7NzTbRSEM2bN4eioiL+/PPPL06CYmxsDA0NjVwxAMDdu3ehoKAACwuLr44hLx8qePHx8YL1+XXPNDMzw9ChQzF06FDExMSgWrVqmDdvHpo3b46bN2/i/v37CAwMRO/evWX7fDoDKgDZowvyuz4jIyNW5YiowDhmjoiIqIC0tLTg7++PmTNnonXr1vm2U1RUhEQiEVR4nj59mufDwTU1NXMlE18rMjISXbp0Qb169bBo0aIC7/dhZs7/zsbp6+sreK2oqIiOHTti586duHXrVq7jfPoYgLxYWFhgwIABOHr0KJYvX55re05ODnx8fPDixQsoKiri119/xd69e/H06VNZm+joaGzZsgX16tWTddv8f+no6MDIyAinT58WrF+1apXgdXZ2tqyb5AclSpSAubm5rAvth+rgp9VAqVSKZcuWCfYzMzODo6MjAgMDBe/7rVu3cPTo0UJ5mDkR/bxYmSMiIvoK+XUz/FTLli2xZMkSNGvWDN27d0dMTAxWrlwJa2tr3LhxQ9C2evXqOHbsGJYsWQJzc3NYWVnlmnr/S0aMGIFXr15h/Pjx2LZtm2BblSpV8u0C6ejoiG7dumHVqlVISEhA3bp1cfz4cTx8+DBX2/nz5+PEiROoXbs2BgwYADs7O8TFxeHq1as4duwY4uLiPhujj48PHj16hBEjRmDXrl1o1aoV9PX1ER4ejr///ht3795F165dAQBz585FUFAQ6tWrh6FDh0JJSQlr1qxBeno6Fi5c+FX35kv69++P+fPno3///qhRowZOnz6N+/fvC9q8ffsWpUqVQqdOneDg4AAtLS0cO3YMly5dgo+PD4B3XUnLlSuHcePG4eXLl9DR0cHOnTvzHLe3aNEiNG/eHE5OTujXr5/s0QS6urpffC4gEZGAPKfSJCIiKso+fTTB5+T1aIINGzZIbWxspKqqqtKKFStKAwIC8nykwN27d6UNGjSQqqurSwHIHlPwoe2rV69yne+/x2nYsKEUQJ7Lp9Pr5yU1NVU6YsQIqaGhoVRTU1PaunVr6fPnz/PcNzo6Wjps2DCphYWFVFlZWWpqaipt0qSJdO3atZ89xwdZWVnS9evXS+vXry/V1dWVKisrS8uUKSPt06dPrscWXL16Verq6irV0tKSamhoSBs3biw9d+6coE1+78+JEyekAKQnTpyQrcvr0QRS6btHSPTr10+qq6sr1dbWlnbp0kUaExMjuP709HSpp6en1MHBQaqtrS3V1NSUOjg4SFetWiU41p07d6QuLi5SLS0tqZGRkXTAgAHS69ev5/n4g2PHjkmdnZ2l6urqUh0dHWnr1q2ld+7cKdB9JCL6QCKVfmF0MBERERERERU5HDNHREREREQkQkzmiIiIiIiIRIjJHBERERERkQgxmSMiIiIiIhIhJnNEREREREQixGSOiIiIiIhIhPjQcCKSG93uf8g7BHovKrCXvEOg9yQSeUdAH7x8kyrvEOg9RQX+xyhKLA3V5HZu9aoehXbs1NAVhXbswsLKHBERERERkQixMkdEREREROIgYS3qU7wbREREREREIsTKHBERERERiQMHFguwMkdERERERCRCrMwREREREZE4cMycAO8GERERERGRCLEyR0RERERE4sAxcwJM5oiIiIiISBzYzVKAd4OIiIiIiEiEWJkjIiIiIiJxYDdLAVbmiIiIiIiIRIiVOSIiIiIiEgeOmRPg3SAiIiIiIhIhVuaIiIiIiEgcOGZOgJU5IiIiIiIiEWJljoiIiIiIxIFj5gSYzBERERERkTiwm6UAU1siIiIiIiIRYmWOiIiIiIjEgd0sBXg3iIiIiIiIRIiVOSIiIiIiEgeOmRNgZY6IiIiIiEiEWJkjIiIiIiJx4Jg5Ad4NIiIiIiIiEWJljoiIiIiIxIGVOQEmc0REREREJA4KnADlU0xtiYiIiIiIRIiVOSIiIiIiEgd2sxTg3SAiIiIiIhIhVuaIiIiIiEgc+NBwAVbmiIiIiIiIRIiVOSIiIiIiEgeOmRPg3SAiIiIiIhIhVuaIiIiIiEgcOGZOgMkcERERERGJA7tZCvBuEBERERERiRArc0REREREJA7sZinAyhwREREREZEIMZkrBpYtW4aQkBB5h0H/p40bN+LQoUPyDoOIiIhIfiQKhbeIkDij/j89fvwYJUuWRJs2bRATE4OqVasWynnc3d3Rrl27Qjl2Qfn4+GDXrl2oVq3aZ9udPHkSEokE8fHxAN4lDnp6eoUfYBHj5+cHfX19+Pv7Y/369Vi0aJG8QwIA7Ny5EwsXLkSdOnXkHQoRERERFRFyT+aioqIwfPhwlC1bFqqqqrCwsEDr1q1x/Pjx//vY+SVTR48exeDBg9GwYUPUrl0bAwcO/L/PVRQFBwfjjz/+wN69e6GqqvpV+/7222+4f/++7PXMmTPh6Oj43WJLTU2FpqYmHj58mG+bEydOoEWLFjA0NISGhgbs7OwwduxYvHz58v8+v0QiwZ49e3Kt37FjBw4ePIgjR45g6dKl6Ny58/99rv/Xw4cPMXXqVBw6dAj6+vryDuenpyCRYEpnB9zwbY+ojd1wbWk7eLa3z9VucicH3FvZEVEbu2HvZBeUNdXO83gqSgo449USCVt6wb4M37/vrfmvv8CxcoVci9fcWfIOrdjatmUzmjf9BTWr2qNH1864eeOGvEP66dy8dgUzx49Az7ZN0aKeI86d/le2LSsrE7+v8sWQ3p3Q3qUOerZtisVzpiL2dYzgGA/vhWHyqEHo3KwefmvREH4LZiM1JeVHX4ro3Qy9gumew9GtjQtc6zrg3Kl/BdvPnjyGSSMHoVOzBnCt64BH9+/mOobnsH5wresgWJYtnPOjLkF8JJLCW0RIrhOgPH36FM7OztDT08OiRYtgb2+PzMxMHDlyBMOGDcPdu7n/wRdEdnY2JJ95QwYPHiz7+9ixY7/pHEWRVCpFdnY2lJTeva3Ozs64du3aNx1LXV0d6urq3zE6oaCgIJQpUwbW1tZ5bl+zZg2GDh0KNzc37Ny5E5aWlggPD8emTZvg4+ODJUuWfNN5MzIyoKKiku/206dPA0Ceid6PlJmZCWVlZQCAtbU1wsLC5BpPcTK6TSX0cymPwf7ncPdFPKqWNcTKQXWRmJKJNUfefSaNal0Jg1wrYsjqYDyLScKUzo7YPbEJannuQ3pmjuB4s7tXQ1R8KqrI42KKgc3bdiAnJ1v2+uGDBxg8oA+a/tpMjlEVX4cPHcTihd6YOmMW7O0dsPmPQAwZ1A979x+GoaGhvMP7aaSlpsLKujx+bdkOc6eMEWxLT0vDw/th6OY2AGVtKiApMRGrly3ErAmj4LdhCwAg9nUMJo8ahAZNXDF0zCSkJCdhjd8iLPGajilzF8vjkkQrLS0VZa0rwLVVO8yeNCb39tRUVHKoigZNXOE7P/8vmZq36YjeA4bKXquqqRVKvPTzkWtlbujQoZBIJLh48SI6duyI8uXLo1KlShgzZgzOnz8va7dkyRLY29tDU1MTFhYWGDp0KJKSkmTbP3QJ3LdvH+zs7KCqqoq+ffsiMDAQe/fuhUQigUQiwcmTJwEAEyZMQPny5aGhoYGyZcti2rRpyMzMFMTm7++PcuXKQUVFBRUqVMAff/zx2WvJzs7GmDFjoKenB0NDQ4wfPx5SqVTQJicnB97e3rCysoK6ujocHBywY8eOzx43PT0dEyZMgIWFBVRVVWFtbY0NGzYA+Ng18tChQ6hevTpUVVVx9uzZAp3n4MGDKF++PNTV1dG4cWM8ffpUsP3TbpYbN27ErFmzcP36ddm93LhxIwAgPDwcbdu2hZaWFnR0dNClSxdER0d/9poAYO/evWjTpk2e2168eIERI0ZgxIgR+P3339GoUSNYWlqiQYMGWL9+PaZPnw4AiI2NRbdu3VCyZEloaGjA3t4eW7duFRyrUaNG8PDwwKhRo2BkZARXV1dYWloCANq3bw+JRCJ7/ejRI7Rt2xYmJibQ0tJCzZo1cezYMcHx3rx5g969e0NfXx8aGhpo3rw5Hjx48NlrlUgk8Pf3R/PmzaGuro6yZcsK3o+nT59CIpHgr7/+QsOGDaGmpobNmzcDANavXw9bW1uoqamhYsWKWLVq1WfP9eHfxIEDB1ClShWoqamhTp06uHXrlqxNQe7b27dv0aNHD2hqasLMzAxLly5Fo0aNMGrUqP/rXohBLRtjHLz8AkevvUT462TsvRiOEzcjUL3cx19EhzSriMV7buLglRe4/Tweg/2DYaqngVY1SguO5eJgjl/szTF185UffRnFhoGBAYyMjGXL6VMnYGFRGjVq1pJ3aMXSH4EB6NCpC9q174hy1taYOmMW1NTUsGfXTnmH9lOp6VQPbgM9ULfhL7m2aWppw8t3DRo0cUWp0paoWLkKho6ZiIf37iAmKhIAcDH4NJSUlDB0zCSUKm2J8raV4TFuKoJPHkPEi/AffTmiVtOpHtwHecC5YZM8t7s0b42efQejas3anz2OqpoaDAyNZIumplZhhPtz4Jg5AblFHRcXh8OHD2PYsGHQ1NTMtf3T8VoKCgrw8/PD7du3ERgYiH///Rfjx48XtE9JScGCBQuwfv163L59G35+fujSpQuaNWuGyMhIREZGom7dugAAbW1tbNy4EXfu3MGyZcuwbt06LF26VHas3bt3Y+TIkRg7dixu3bqFQYMGoU+fPjhx4kS+1+Pj44ONGzfi999/x9mzZxEXF4fdu3cL2nh7e2PTpk1YvXo1bt++jdGjR6Nnz544depUvsft3bs3tm7dCj8/P4SFhWHNmjXQ0hL+B584cSLmz5+PsLAwVKlS5Yvnef78OTp06IDWrVvj2rVr6N+/PyZOnJhvDL/99hvGjh2LSpUqye7lb7/9hpycHLRt2xZxcXE4deoUgoKC8PjxY/z222/5Hgt4l9Tu378fbdu2zXP733//jYyMjFzv8Qcf/m2kpaWhevXqOHDgAG7duoWBAweiV69euHjxoqB9YGAgVFRUEBwcjNWrV+PSpUsAgICAAERGRspeJyUloUWLFjh+/DhCQ0PRrFkztG7dGuHhH3+wubu74/Lly9i3bx9CQkIglUrRokWLXF8G/Ne0adPQsWNHXL9+HT169EDXrl1zVdsmTpyIkSNHIiwsDK6urti8eTOmT5+OefPmISwsDF5eXpg2bRoCAwM/ey4A8PT0hI+PDy5dugRjY2O0bt1aFmNB7tuYMWMQHByMffv2ISgoCGfOnMHVq1cF5/jWe1HUXXzwCg0qm6Lc+26TlUvro06FEgi6HgEAsCyhBVN9DZy8FSnbJzE1E5cfvUZNGyPZOmMdNfj1r4NBq84iNT3rx15EMZWZmYGD+/ehbfuOn+2dQYUjMyMDYXduo45TXdk6BQUF1KlTFzeuh8oxMkpOSoJEIoGW9rvPtczMTCgpK0NB4eOvgR+GY9y+wfdKHk4cPYjOzRtiYI8O+N1/GdLSUuUdUtHFbpYCcutm+fDhQ0ilUlSsWPGLbT+tBlhaWmLu3LkYPHiwoEqRmZmJVatWwcHBQbZOXV0d6enpMDU1FRxv6tSpguONGzcO27ZtkyUPixcvhru7O4YOfVfu/lApXLx4MRo3bpxnjL6+vpg0aRI6dOgAAFi9ejWOHDki256eng4vLy8cO3YMTk5OAICyZcvi7NmzWLNmDRo2bJjrmPfv38f27dsRFBQEFxcX2T7/NXv2bDRt2rTA5/lQdfTx8QEAVKhQATdv3sSCBQvyvDZ1dXVoaWlBSUlJcC+DgoJw8+ZNPHnyBBYWFgCATZs2oVKlSrh06RJq1qyZ5/E+VF1r1877W6oHDx5AR0cHZmZmeW7/oGTJkhg3bpzs9fDhw3HkyBFs374dtWp9/FbexsYGCxcuzLW/np6e4HocHBwE/37mzJmD3bt3Y9++ffDw8MCDBw+wb98+BAcHy74Y2Lx5MywsLLBnz57Pjq/r3Lkz+vfvLztuUFAQli9fLvg3PGrUKNm/HwCYMWMGfHx8ZOusrKxw584drFmzBm5ubp+9NzNmzJD9mwgMDESpUqWwe/dudOnS5Yv37e3btwgMDMSWLVvQpMm7bxoDAgJgbm4u2+db7kV6ejrS09MF66TZmZAoKn/2Wn60JftuQVtdGZcXt0V2jhSKChLM2X4Nfwc/AQCU0H3X/TgmIU2w36uEVJjofuya7D+4Ln4//gChT+JQ2ij3F1b0/f17/Bjevn2LNu3ayzuUYulN/BtkZ2fn6k5paGiIJ08eyykqykhPR4D/MjR0aQaN99Ueh2o1sW65D3Zs2Yi2nXsgLTUVAav9AABxsa/lGW6x1Lhpc5QwNYOhcQk8eXgfG1b54kX4U0z3XvrlnanYk1sy998uiJ9z7NgxeHt74+7du0hMTERWVhbS0tKQkpICDQ0NAICKigqqVCnYqJS//voLfn5+ePToEZKSkpCVlQUdHR3Z9rCwsFyTojg7O2PZsmV5Hi8hIQGRkZGC5ERJSQk1atSQXefDhw+RkpIi+wX7g4yMjHxn07x27RoUFRXzTPQ+VaNGDdnfC3KesLCwXInUh8Tva4SFhcHCwkKWyAGAnZ0d9PT0EBYWlm8yt3fvXrRq1UrwjeCnpFJpgb5Vz87OhpeXF7Zv346XL18iIyMD6enpsn8TH1SvXr1A15OUlISZM2fiwIEDiIyMRFZWFlJTU2WVubCwMCgpKQnunaGhISpUqPDFMW3/vb9OTk65xjN++j4mJyfj0aNH6NevHwYMGCBbn5WVBV1dXQBA8+bNcebMGQBAmTJlcPv27TzPZ2BgIIjxS/ft8ePHyMzMFCTEurq6qFChguz1t9wLb29vzJolHC+gUrkd1Ow75NleXjrUsURnZyv0X3kWYS/iYV9GH/N71UTkmxRsPVOwX0gHuVaElroyluy99eXG9N3s2bUTzvUaoEQJE3mHQlQkZGVlwnv6eEghhce4KbL1ZcpaY8yU2Vi/wgcb1yyHgoIC2nbqBn0DQyiItKuZmLVo10n2d6tyNjAwNMKEEQMR8eI5zEtZfGbPYor/RgXklszZ2NhAIpF8cZKTp0+folWrVhgyZAjmzZsHAwMDnD17Fv369UNGRobsF1B1dfUCJQAhISHo0aMHZs2aBVdXV+jq6mLbtm2yKlVh+TDG78CBAyhZsqRgW34zTRZ0ApJPu6l+y3l+tH379mH+/Pn5bi9fvrwsQf5cdW7RokVYtmwZfH19ZWMqR40ahYyMDEG7vLrx5mXcuHEICgrC4sWLYW1tDXV1dXTq1CnX8QpLXu/junXrciXeioqKAN6Np0tNfdcN48NkKQVR0Pv2vU2aNAljxggHh5ca8Pkxo/Iwu3s1LN13CztDngIA7jyPh4WRFsa0rYytZx4jJuHdPS+hq4bo+I/dYIx11XHzWRwAoEElU9SyMcKrTd0Fxz45twW2Bz/BkNXnfszFFCMRES9x4fw5+Pgul3coxZa+nj4UFRURGxsrWB8bGwsjI6N89qLCkpWVCe9p4xETFQlvv7WyqtwHjX9tgca/tsCbuFioqb37HWr3X3/C1LxkPkekH6VipXczKEe8CGcyR18kt9TWwMAArq6uWLlyJZKTk3Nt//C8sytXriAnJwc+Pj6oU6cOypcvj4iIiAKdQ0VFBdnZ2YJ1586dQ5kyZTBlyhTUqFEDNjY2ePbsmaCNra0tgoODBeuCg4NhZ2eX53l0dXVhZmaGCxcuyNZlZWXhypWPkx58mJglPDwc1tbWguXTytan7O3tkZOT89kxdf9VkPPY2trmGlf26YQzecnrXtra2uL58+d4/vy5bN2dO3cQHx+f77168OABnj17lqty+KlOnTpBRUUlz66RwMd/G8HBwWjbti169uwJBwcHlC1bVvA4hc9RVlbOdT3BwcFwd3dH+/btYW9vD1NTU8HEMLa2tsjKyhK8z7Gxsbh3716+1/vBf+/v+fPnYWtrm297ExMTmJub4/Hjx7neRysrKwDvupl+WFemTJl8z/fmzRvcv39fdr4v3beyZctCWVlZNpYQeFd9/rTNt9wLVVVV6OjoCJai1sUSADRUlPDfjgM5OVIovP+y6GlMEqLepKBhpY9ddLXVlVGjnBEuPXjXPWlC4EU4TzyAepPeLZ0Xvpuquo/fGczZfu2HXEdxs3f3LhgYGKJ+g0byDqXYUlZRga1dJVw4HyJbl5OTgwsXQlDFoXCe50p5+5DIRbwIh5fvaujo6uXbVt/AEOoaGjh9/AiUVVRQtSafZypvjx7cAwAYGBnLOZIiihOgCMj10QQrV66Es7MzatWqhdmzZ6NKlSrIyspCUFAQ/P39ERYWBmtra2RmZmL58uVo3bq1bBKLgrC0tMSRI0dw7949GBoaQldXFzY2NggPD8e2bdtQs2ZNHDhwINdEJZ6enujSpQuqVq0KFxcX/PPPP9i1a1eumQ0/NXLkSMyfPx82NjaoWLEilixZIks6gHeTrowbNw6jR49GTk4O6tWrh4SEBAQHB0NHRyfPMVCWlpZwc3ND37594efnBwcHBzx79gwxMTHo0qVLnnEU5DyDBw+Gj48PPD090b9/f1y5ckU2O+Xn7uWTJ09w7do1lCpVCtra2nBxcYG9vT169OgBX19fZGVlYejQoWjYsKGgy+Cn9u7dCxcXl1xdIT9lYWGBpUuXwsPDA4mJiejduzcsLS3x4sULbNq0CVpaWvDx8YGNjQ127NiBc+fOQV9fH0uWLEF0dPQXE6sP13P8+HE4OztDVVUV+vr6sLGxwa5du9C6dWtIJBJMmzYNOTkfp5m3sbFB27ZtMWDAAKxZswba2tqYOHEiSpYsme9kLh/8/fffqFGjBurVq4fNmzfj4sWLsllJ8zNr1iyMGDECurq6aNasGdLT03H58mW8efMmV4Xrv2bPng1DQ0OYmJhgypQpMDIykj1z8Uv3TVtbG25ubvD09ISBgQFKlCiBGTNmQEFBQVb9/n/uRVF36OoLjG1bGc9fJ+Pui3hUsTTAsBa2+PPkx2ci+h++C8/29ngU9RbPXr17NEFUfAr2X37XJfdFbAqAj89rSk57NynMk5i3iIjjc5y+t5ycHOzbswut27aTPZqF5KOXWx9MmzwBlSpVRmX7Kvjzj0CkpqaiXfui1Z1a7FJTUhDx8uPkXNGRL/HowV1oa+vCwMgIXlM98fB+GGYu8EN2To5sHJy2jq6sJ8c/O7fBtrID1NQ1EHopBL+v8oX74BHQ0tbJ85yUt9SUFMEMoFGRL/Ho/l1o6+iihKkZEhMT8CoqErGvXwEAnoc/BQDov5+1MuLFc5wIOohaTvWhrauLJw8fYM2yRbB3rI6y1uXlcUkkMnL9qVe2bFlcvXoV8+bNw9ixYxEZGQljY2NUr14d/v7+AN5NSrFkyRIsWLAAkyZNQoMGDeDt7Y3evXt/8fgDBgzAyZMnUaNGDSQlJeHEiRNo06YNRo8eDQ8PD6Snp6Nly5aYNm0aZs6cKduvXbt2WLZsGRYvXoyRI0fCysoKAQEBaNSoUb7n+hC/m5sbFBQU0LdvX7Rv3x4JCQmyNnPmzIGxsTG8vb3x+PFj6OnpoVq1apg8eXK+x/X398fkyZMxdOhQxMbGonTp0p9tX5DzlC5dGjt37sTo0aOxfPly1KpVC15eXujbt2++x+zYsSN27dqFxo0bIz4+HgEBAXB3d8fevXsxfPhwNGjQAAoKCmjWrBmWL8+/m9PevXu/OHkH8O6xFeXLl8fixYvRvn17pKamwtLSEq1atZIlMlOnTsXjx4/h6uoKDQ0NDBw4EO3atRPc8/z4+PhgzJgxWLduHUqWLImnT59iyZIl6Nu3L+rWrQsjIyNMmDABiYmJgv0CAgIwcuRItGrVChkZGWjQoAEOHjz4xW6Os2bNwrZt2zB06FCYmZlh69atX0w6+/fvDw0NDSxatAienp7Q1NSEvb29YEKg/MyfPx8jR47EgwcP4OjoiH/++Uf2fL2C3LclS5Zg8ODBaNWqFXR0dDB+/Hg8f/4cap889+Zb70VRNz7wIqZ0doRPn1ow1lVD1JtUBBx/gAW7Pj742Pef29BQVcKy/nWgq6GC8/dj0GH+8VzPmKMf43zIOURGRqBd+47yDqXYa9a8Bd7ExWHVCj+8fv0KFSraYtWa9TBkN8vv6sHd25g44uN46nXL3w0VcWneGj36Dsb5sycBAB59hLNLz/dbhyrV3o1nv3fnFv7c4I/U1BRYlLaCh+dUNGnW6sdcwE/k/t3bGO/RX/Z6jd+75/Q1bdEG46bOwfkzJ+Ezb7psu/f0CQCAnn0Ho1f/IVBSVkbopQvY/ddmpKWlwriEKeo1dkE39wGgfIh01snCIpF+zUwkRP+H169fw8zMDC9evICJSfGZoEAikWD37t2yylhhOnnyJBo3bow3b94IHu/x/0pOTkbJkiXh4+ODfv36fbfj6nb//PMb6ceJCuwl7xDoPf6eUnS8fMPp4YsKRQX+xyhKLA3l91Bz9Tb+hXbs1H1DCu3YhYX9UeiHiYuLw5IlS4pVIidWoaGhuHv3LmrVqoWEhATMnj0bAETfhZKIiIhETqRj2woLkzn6YcqXL4/y5dn/WywWL16Me/fuQUVFBdWrV8eZM2c4Ix0RERHJF7svCDCZIypkP7Inc6NGjb7L+apWrSqYjZWIiIiIih4mc0REREREJA7sZinAu0FERERERCRCrMwREREREZE4cMycACtzREREREREIsTKHBERERERiYKElTkBVuaIiIiIiIhEiJU5IiIiIiISBVbmhFiZIyIiIiIiEiFW5oiIiIiISBxYmBNgMkdERERERKLAbpZC7GZJREREREQkQqzMERERERGRKLAyJ8TKHBERERERkQixMkdERERERKLAypwQK3NEREREREQixGSOiIiIiIhEQSKRFNpSUNnZ2Zg2bRqsrKygrq6OcuXKYc6cOZBKpbI2UqkU06dPh5mZGdTV1eHi4oIHDx4IjhMXF4cePXpAR0cHenp66NevH5KSkr7qfjCZIyIiIiIiKqAFCxbA398fK1asQFhYGBYsWICFCxdi+fLlsjYLFy6En58fVq9ejQsXLkBTUxOurq5IS0uTtenRowdu376NoKAg7N+/H6dPn8bAgQO/KhaOmSMiIiIiInEoAkPmzp07h7Zt26Jly5YAAEtLS2zduhUXL14E8K4q5+vri6lTp6Jt27YAgE2bNsHExAR79uxB165dERYWhsOHD+PSpUuoUaMGAGD58uVo0aIFFi9eDHNz8wLFwsocERERERGJQmF2s0xPT0diYqJgSU9PzxVD3bp1cfz4cdy/fx8AcP36dZw9exbNmzcHADx58gRRUVFwcXGR7aOrq4vatWsjJCQEABASEgI9PT1ZIgcALi4uUFBQwIULFwp8P5jMERERERFRseft7Q1dXV3B4u3tnavdxIkT0bVrV1SsWBHKysqoWrUqRo0ahR49egAAoqKiAAAmJiaC/UxMTGTboqKiUKJECcF2JSUlGBgYyNoUBLtZEhERERGRKBTmowkmTZqEMWPGCNapqqrmard9+3Zs3rwZW7ZsQaVKlXDt2jWMGjUK5ubmcHNzK7T48sJkjoiIiIiIij1VVdU8k7f/8vT0lFXnAMDe3h7Pnj2Dt7c33NzcYGpqCgCIjo6GmZmZbL/o6Gg4OjoCAExNTRETEyM4blZWFuLi4mT7FwS7WRIRERERkSgUhUcTpKSkQEFBmEYpKioiJycHAGBlZQVTU1McP35ctj0xMREXLlyAk5MTAMDJyQnx8fG4cuWKrM2///6LnJwc1K5du8CxsDJHRERERERUQK1bt8a8efNQunRpVKpUCaGhoViyZAn69u0L4F3COWrUKMydOxc2NjawsrLCtGnTYG5ujnbt2gEAbG1t0axZMwwYMACrV69GZmYmPDw80LVr1wLPZAkwmSMiIiIiIpEozDFzBbV8+XJMmzYNQ4cORUxMDMzNzTFo0CBMnz5d1mb8+PFITk7GwIEDER8fj3r16uHw4cNQU1OTtdm8eTM8PDzQpEkTKCgooGPHjvDz8/uqWCTSTx9VTkT0A+l2/0PeIdB7UYG95B0CvVcEfk+h916+SZV3CPSeogL/YxQlloZqX25USAx7by20Y8du6lZoxy4srMwREREREZE4MK8XYDJHRERERESiUBS6WRYlnM2SiIiIiIhIhFiZIyIiIiIiUWBlToiVOSIiIiIiIhFiZY6IiIiIiESBlTkhVuaIiIiIiIhEiJU5IiIiIiISBxbmBFiZIyIiIiIiEiFW5oiIiIiISBQ4Zk6IyRwREREREYkCkzkhJnNEJDcP13STdwj0Xp8tofIOgd7b2KOqvEOg90x01OQdAr0Xn5Ih7xCIiiQmc0REREREJAqszAlxAhQiIiIiIiIRYmWOiIiIiIhEgZU5IVbmiIiIiIiIRIiVOSIiIiIiEgcW5gRYmSMiIiIiIhIhVuaIiIiIiEgUOGZOiMkcERERERGJApM5IXazJCIiIiIiEiFW5oiIiIiISBRYmRNiZY6IiIiIiEiEWJkjIiIiIiJxYGFOgJU5IiIiIiIiEWJljoiIiIiIRIFj5oRYmSMiIiIiIhIhVuaIiIiIiEgUWJkTYjJHRERERESiwGROiN0siYiIiIiIRIiVOSIiIiIiEgVW5oRYmSMiIiIiIhIhVuaIiIiIiEgcWJgTYGWOiIiIiIhIhFiZIyIiIiIiUeCYOSFW5oiIiIiIiESIlTkiIiIiIhIFVuaEmMwREREREZEoMJcTYjdLIiIiIiIiEWJljoiIiIiIRIHdLIVYmSMiIiIiIhIhVuaIiIiIiEgUWJgTYmWOiIiIiIhIhFiZIyIiIiIiUeCYOSFW5oiIiIiIiESIlTkiIiIiIhIFFuaEmMwREREREZEoKCgwm/sUu1kSERERERGJECtzREREREQkCuxmKcTKHBERERERkQgxmSMqYpYtW4aQkBB5h0FERERU5EgkkkJbxIjJ3E9m48aN0NPTk3cY9I18fHywa9cuVKtW7f86jru7O9q1a/d9giIiIiKiIolj5kTC3d0d8fHx2LNnj2D9yZMn0bhxY7x58wZ6enr47bff0KJFi+967qdPn8LKygqhoaFwdHT8rscuDKdOnULPnj3x/PnzXNs+3K+8REZGwtTUtLDDy1dwcDD++OMPnDx5EqqqqgXaJ7/3ZtmyZZBKpYUUafHzKiYa/suX4Py5M0hLS0OpUqUxecZcVLSrDACYN3MyDu3fK9inlpMzlixfK49wfxqdHEzRydFMsO5lQhrG7gkDAPSvYwF7c23oqysjLSsb92OSseVKBCIS02Xtt7lVzXXcZaeeIORpfKHGXpxt27IZgQEb8Pr1K5SvUBETJ0+DfZUq8g7rp/b7+jU4cTwIT588hqqqGqo4VsWIUWNhaVVW1mbXjr9w+OB+3A27g+TkZJw8exHaOjpyjPrnlZKcjIC1K3D21L+IfxMH6/IVMWz0BNnPjCZ18v7/MNBjNH7r2edHhipKIi2gFRomcz8ZdXV1qKuryzsMudq7dy9at2792Tb37t2Dzn9+iJUoUaIww8pFKpUiOzsbSkrv/hs6Ozvj2rVr3+XYurq63+U4BCQmJmBIv56oVqMWFi9bDT19A7x4/izXL0G169bD5OlzZa+VVVR+dKg/pedvUjH36EPZ65xPvqR4EpuCs0/iEJuUCU1VRXRyNMPkptYYvus2Pv0uw//sM1x7mSh7nZKR/UNiL44OHzqIxQu9MXXGLNjbO2DzH4EYMqgf9u4/DENDQ3mH99O6evkSOnftjkqV7JGdnY0VfksxbHB/7Ni9H+oaGgCAtNQ0ODnXh5NzfaxYtkTOEf/cfLxm4snjh5g0Yx4MjUrg2OH9GD98IDZs3Q3jEib4+8C/gvYXQ85i8bwZqN+4qZwiJjFjN8ufzH+7Wc6cOROOjo5Ys2YNLCwsoKGhgS5duiAhIUHWJicnB7Nnz0apUqWgqqoKR0dHHD58WLbdysoKAFC1alVIJBI0atRItm39+vWwtbWFmpoaKlasiFWrVgniefHiBbp16wYDAwNoamqiRo0auHDhAgDg0aNHaNu2LUxMTKClpYWaNWvi2LFjgv1XrVoFGxsbqKmpwcTEBJ06dfriPdi3bx/atGnz2TYlSpSAqampYFFQePff4UMXRS8vL5iYmEBPTw+zZ89GVlYWPD09YWBggFKlSiEgIEB2vKdPn0IikWDbtm2oW7cu1NTUULlyZZw6dUrW5uTJk5BIJDh06BCqV68OVVVVnD17Fjk5OfD29oaVlRXU1dXh4OCAHTt2yPZ78+YNevToAWNjY6irq8PGxkZ27vzem/92s2zUqBGGDx+OUaNGQV9fHyYmJli3bh2Sk5PRp08faGtrw9raGocOHRLcp1OnTqFWrVpQVVWFmZkZJk6ciKysLNn2HTt2wN7eHurq6jA0NISLiwuSk5O/+B6JyebADShhYorJM+bBrnIVmJcshVp1nFGyVGlBOxVlFRgaGcsWHR0m1N9DtlSKhLQs2fI2/WMidvxBLO5GJ+NVcgaexqVie2gEjLRUUEJLmEgnZ2QLjpGZw6p1YfkjMAAdOnVBu/YdUc7aGlNnzIKamhr27Nop79B+aitWr0ebth1QztoG5StUxKw53oiKjEDYnduyNt17uaFPv4Gwr+Igx0h/fulpaTh98hgGeoxGlao1UNKiNNwGDIV5KQv8s2s7AMDA0EiwBJ8+AcfqNWFespScoxcHjpkTYjJXDDx8+BDbt2/HP//8g8OHDyM0NBRDhw6VbV+2bBl8fHywePFi3LhxA66urmjTpg0ePHgAALh48SIA4NixY4iMjMSuXbsAAJs3b8b06dMxb948hIWFwcvLC9OmTUNgYCAAICkpCQ0bNsTLly+xb98+XL9+HePHj0dOTo5se4sWLXD8+HGEhoaiWbNmaN26NcLDwwEAly9fxogRIzB79mzcu3cPhw8fRoMGDT57rbdv30ZMTAx++eWX/+ue/fvvv4iIiMDp06exZMkSzJgxA61atYK+vj4uXLiAwYMHY9CgQXjx4oVgP09PT4wdOxahoaFwcnJC69atERsbK2gzceJEzJ8/H2FhYahSpQq8vb2xadMmrF69Grdv38bo0aPRs2dPWSI4bdo03LlzB4cOHUJYWBj8/f1hZGQEIP/3Ji+BgYEwMjLCxYsXMXz4cAwZMgSdO3dG3bp1cfXqVfz666/o1asXUlJSAAAvX75EixYtULNmTVy/fh3+/v7YsGED5s59V32KjIxEt27d0LdvX4SFheHkyZPo0KHDT9e9M/j0CVS0rYSpE0ajVdP66NO9I/bt/jtXu9Arl9CqaX1069ASi71nIyE+/scH+xMy1VbFqs6VsayDHTzql4GhpnKe7VSVFNDI2hDRb9PxOjlTsK1vnVJY+5s95rYsj0bWBj8i7GIpMyMDYXduo45TXdk6BQUF1KlTFzeuh8oxsuInKektAECHvTR+uOzsbORkZ0PlP70zVFXVcCuP/wdxsbG4EHwGzVu3/1Eh0k+G3SxFZP/+/dDS0hKsy87+cnehtLQ0bNq0CSVLlgQALF++HC1btoSPjw9MTU2xePFiTJgwAV27dgUALFiwACdOnICvry9WrlwJY2NjAIChoaFgTNmMGTPg4+ODDh06AHhXJbpz5w7WrFkDNzc3bNmyBa9evcKlS5dgYPDuFyhra2vZ/g4ODnBw+PgN4Zw5c7B7927s27cPHh4eCA8Ph6amJlq1agVtbW2UKVMGVavmHv/yqb1798LV1TXXh+h/lSol/ParTJkyuH374zeYBgYG8PPzg4KCAipUqICFCxciJSUFkydPBgBMmjQJ8+fPx9mzZ2X3DQA8PDzQsWNHAIC/vz8OHz6MDRs2YPz48bI2s2fPRtOm77pSpKenw8vLC8eOHYOTkxMAoGzZsjh79izWrFmDhg0bIjw8HFWrVkWNGjUAAJaWlrJj5ffe5MXBwQFTp04VxG9kZIQBAwYAAKZPnw5/f3/cuHEDderUwapVq2BhYYEVK1ZAIpGgYsWKiIiIwIQJEzB9+nRERkYiKysLHTp0QJkyZQAA9vb2+Z4/PT0d6enpwnUZigUeHygvES9fYM/Ov/BbDzf07jMQYXduwnexN5SVldG8VTsAQG2nemjY2AVmJUvh5YvnWLvSF+NGDMLqgC1QVFSU7wWI2MPXKfAPDkdkYhr01JXRycEUM5uVh+feMKRlvftSqGkFI/Sobg41ZUW8TEiDV9BDZH9SedseGoFbkUnIyM5BFXNt9K1jATUlRRy++0pel/XTehP/BtnZ2bm6UxoaGuLJk8dyiqr4ycnJweKFXnCoWg3WNuXlHU6xo6GpCTt7B/z5+1qUtiwLfQND/Hv0EO7cug7zUha52h89uBcamhqo38hFDtGKk1graIWFyZyING7cGP7+/oJ1Fy5cQM+ePT+7X+nSpWWJHAA4OTkhJycH9+7dg4aGBiIiIuDs7CzYx9nZGdevX8/3mMnJyXj06BH69esnSwYAICsrSzZe69q1a6hataoskfuvpKQkzJw5EwcOHJAlBqmpqbLKXNOmTVGmTBmULVsWzZo1Q7NmzdC+fXtovO//n5e9e/fCw8Pjs/cDAM6cOQNtbW3Za2Vl4bf9lSpVknW7BAATExNUrlxZ9lpRURGGhoaIiYkR7PchIQMAJSUl1KhRA2FhYYI2H5Iy4F3VNCUlRZbcfZCRkSFLXIcMGYKOHTvKqmft2rVD3bp18bWqfDIBwYf4P02+TExMAEB2TWFhYXBychJ8aDo7OyMpKQkvXryAg4MDmjRpAnt7e7i6uuLXX39Fp06doK+vn+f5vb29MWvWLMG6cROnYfzk6V99LT9STk4OKtpVxqBhowAA5Sva4smjh9izc7ssmXNx/TjpUDnr8ihnXR6/tWuG0CuXUKNWHTlE/XP4dJxb+Js0PHyVghWdKsHJUg8nHsYBAM4+jsPNiLfQ01BCq0omGNnQCjMO3pd1pdx1I1p2jKdxqVBVUkDryiWYzNFPa/682Xj08AE2bNwi71CKrUkzvLBo3nT81toFCoqKsKlgi8ZNm+PB3Tu52h7evwdNfm0JlSL+xWZRwlxOiMmciGhqagoqWwBydfP7UZKSkgAA69atQ+3atQXbPlQivjQRy7hx4xAUFITFixfD2toa6urq6NSpEzIyMgAA2trauHr1Kk6ePImjR49i+vTpmDlzJi5dupTn4xciIyMRGhqKli1bfjF+Kyurzz7C4b/JnUQiyXPdhy6jX0NTU1P29w/38cCBA4KEG4CsYtW8eXM8e/YMBw8eRFBQEJo0aYJhw4Zh8eLFX3XeL13Th6StoNekqKiIoKAgnDt3DkePHsXy5csxZcoUXLhwQTaW71OTJk3CmDFjBOsSM4p+1crQyBiWVuUE68pYlcXJf4Py3adkKQvo6enjxfNwJnPfUUpmNiIT02Ci8/GXntTMHKRmpiPqbToevHqCDV3tUbOMHs49eZPnMR6+SkFHBzMoKUiQxbFz35W+nj4UFRVzdS2PjY2VdQ2nwrXAazbOnj6JdQF/wkSOszMXd+alLLDUPwCpqSlISU6GoZEx5kzxhNl/xsTduHYFz589xbS5i+QUKf0MOGauGAgPD0dERITs9fnz52XdB3V0dGBubo7g4GDBPsHBwbCzswMAWZfFT7t0mpiYwNzcHI8fP4a1tbVg+fCLfJUqVXDt2jXExcXlGVdwcDDc3d3Rvn172Nvbw9TUFE+fPhW0UVJSgouLCxYuXIgbN27g6dOn+Pfff/M83j///IO6devmWwn8Ec6fPy/7e1ZWFq5cuQJbW9t829vZ2UFVVRXh4eG57qOFxcfuGMbGxnBzc8Off/4JX19frF37bsr7vN6b78XW1hYhISGCMXDBwcHQ1taWdVOVSCRwdnbGrFmzEBoaChUVFezevTvP46mqqkJHR0ewFPUulgBg71AV4c+eCNY9f/YUpmbm+e4TEx2FhIR4/gL7nakqKcBEWxXxKVl5bpfg/ZcUCvl/bVvGQB1J6VlM5AqBsooKbO0q4cL5ENm6nJwcXLgQgioOn+8iT/8fqVSKBV6zceLfY1i9fiNKluJEGkWBuroGDI2M8TYxEZcunEPdBsJHIx3atxvlK9qhnE0FOUUoTpwARYiVuWJATU0Nbm5uWLx4MRITEzFixAh06dJFNsbK09MTM2bMQLly5eDo6IiAgABcu3YNmzdvBvBu5kd1dXUcPnwYpUqVgpqaGnR1dTFr1iyMGDECurq6aNasGdLT03H58mW8efMGY8aMQbdu3eDl5YV27drB29sbZmZmCA0Nhbm5OZycnGBjY4Ndu3ahdevWkEgkmDZtmqAqtH//fjx+/BgNGjSAvr4+Dh48iJycHFSokPeHXkFmsfwgJiYGaWlpgnWGhoa5qldfa+XKlbCxsYGtrS2WLl2KN2/eoG/fvvm219bWxrhx4zB69Gjk5OSgXr16SEhIQHBwMHR0dODm5obp06ejevXqqFSpEtLT07F//35Zgpjfe/M9DB06FL6+vhg+fDg8PDxw7949zJgxA2PGjIGCggIuXLiA48eP49dff0WJEiVw4cIFvHr16rPJqxj91r03BvftiU2/r8UvTV1x5/ZN7Nu9A+OnzAQApKQkI2CdPxr+0hSGhkZ4+eI5Vvn5oKRFadRyqiff4EWuZw1zXHmeiNdJGdDXUEYnR1PkSKUIfvIGJbRU4GSpjxsRiUhMz4Khhgra2JsgIysHoe+7Z1YrpQNddWU8eJWMzOwcVDHXQTt7E+y/HfOFM9O36uXWB9MmT0ClSpVR2b4K/vwjEKmpqWjXvoO8Q/upzZ83G4cP7ceSZSuhoamJ16/fdSPW0tKGmpoaAOD161eIff0az98PZXj44D40NDVhamYGXV09eYX+U7p0PhhSqRQWZSzx8vlzrF2xBKXLWKJZq7ayNsnJSTj971EMHjFOjpHSz4DJXDFgbW2NDh06oEWLFoiLi0OrVq0EjxAYMWIEEhISMHbsWMTExMDOzg779u2DjY0NgHfVMT8/P8yePRvTp09H/fr1cfLkSfTv3x8aGhpYtGgRPD09oampCXt7e4waNQrAu6rR0aNHMXbsWLRo0QJZWVmws7PDypUrAQBLlixB3759UbduXRgZGWHChAlITPw4RkZPTw+7du3CzJkzkZaWBhsbG2zduhWVKlXKdY3Jyck4fvw4fH19C3RP8koIQ0JCUKfO/9clbv78+Zg/fz6uXbsGa2tr7Nu374vVmTlz5sDY2Bje3t54/Pgx9PT0UK1aNdlkKyoqKpg0aRKePn0KdXV11K9fH9u2bQOQ/3vzPZQsWRIHDx6Ep6cnHBwcYGBggH79+skmUdHR0cHp06fh6+uLxMRElClTBj4+PmjevPl3OX9RYVvJHl6Ll2HNCl9sXO8PM/NSGDF2An5t3goAoKigiEcP7uHQ/r1IepsII+MSqFmnLgYMHv7FiXjo8ww0VDC8gSW0VRWRmJaFezHJmHbwPt6mZ0FJQQkVTTTR3M4YWiqKSEjLQlh0EqYfuo/EtHeVu+wcKX6tYITeNUtCAiDqbTr+uPwS/96P/fyJ6Zs1a94Cb+LisGqFH16/foUKFW2xas16GLJKXah2bN8KABjYt7dg/Yw5XmjT9l0ivXP7NqxdvVK2rX+fnrna0PeRnJSE9f7L8DomGto6uqjf2AV9Bw+HktLHL4xPBB2GVAo0/vXn+pn5I4i0gFZoJNKfbR5xEpg5cyb27Nnz3R5GXVTt2rULU6dOxZ07uQcX/whPnz6FlZUVQkND4ejoKJcYxOjV27y7y9GPN3zXTXmHQO9t7MEuiUVFVjZ/RSoq4lMy5B0CfaKUvvyGSVSbnfdwm+/h6vT/79FW8sDKHP0UtLS0sGDBAnmHQURERESFSKxj2woLkzn6Kfz666/yDoGIiIiI6IfibJY/uZkzZ/70XSyLAktLS0ilUnaxJCIiIipEEknhLWLEyhwREREREYkCu1kKsTJHREREREQkQqzMERERERGRKLAwJ8TKHBERERERkQixMkdERERERKLAMXNCrMwRERERERGJECtzREREREQkCizMCbEyR0REREREJEKszBERERERkShwzJwQkzkiIiIiIhIF5nJC7GZJREREREQkQqzMERERERGRKLCbpRArc0RERERERCLEyhwREREREYkCC3NCrMwRERERERGJECtzREREREQkChwzJ8TKHBERERERkQixMkdERERERKLAypwQkzkiIiIiIhIF5nJC7GZJREREREQkQkzmiIiIiIhIFCQSSaEtX+Ply5fo2bMnDA0Noa6uDnt7e1y+fFm2XSqVYvr06TAzM4O6ujpcXFzw4MEDwTHi4uLQo0cP6OjoQE9PD/369UNSUtJXxcFkjoiIiIiIqIDevHkDZ2dnKCsr49ChQ7hz5w58fHygr68va7Nw4UL4+flh9erVuHDhAjQ1NeHq6oq0tDRZmx49euD27dsICgrC/v37cfr0aQwcOPCrYuGYOSIiIiIiEoXCHDOXnp6O9PR0wTpVVVWoqqoK1i1YsAAWFhYICAiQrbOyspL9XSqVwtfXF1OnTkXbtm0BAJs2bYKJiQn27NmDrl27IiwsDIcPH8alS5dQo0YNAMDy5cvRokULLF68GObm5gWKmZU5IiIiIiIq9ry9vaGrqytYvL29c7Xbt28fatSogc6dO6NEiRKoWrUq1q1bJ9v+5MkTREVFwcXFRbZOV1cXtWvXRkhICAAgJCQEenp6skQOAFxcXKCgoIALFy4UOGYmc0REREREJAqFOWZu0qRJSEhIECyTJk3KFcPjx4/h7+8PGxsbHDlyBEOGDMGIESMQGBgIAIiKigIAmJiYCPYzMTGRbYuKikKJEiUE25WUlGBgYCBrUxDsZklERERERMVeXl0q85KTk4MaNWrAy8sLAFC1alXcunULq1evhpubW2GHKcDKHBERERERiYJEUnhLQZmZmcHOzk6wztbWFuHh4QAAU1NTAEB0dLSgTXR0tGybqakpYmJiBNuzsrIQFxcna1MQTOaIiIiIiEgUFCSSQlsKytnZGffu3ROsu3//PsqUKQPg3WQopqamOH78uGx7YmIiLly4ACcnJwCAk5MT4uPjceXKFVmbf//9Fzk5Oahdu3aBY2E3SyIiIiIiogIaPXo06tatCy8vL3Tp0gUXL17E2rVrsXbtWgDvxvWNGjUKc+fOhY2NDaysrDBt2jSYm5ujXbt2AN5V8po1a4YBAwZg9erVyMzMhIeHB7p27VrgmSwBJnNERERERCQShflogoKqWbMmdu/ejUmTJmH27NmwsrKCr68vevToIWszfvx4JCcnY+DAgYiPj0e9evVw+PBhqKmpydps3rwZHh4eaNKkCRQUFNCxY0f4+fl9VSwSqVQq/W5XRkT0FV69zZJ3CPTe8F035R0CvbexR1V5h0DvZWXzV6SiIj4lQ94h0CdK6X95kpDC8uvK84V27KPD6hTasQsLK3NERERERCQKkqJQmitCOAEKERERERGRCLEyR0REREREoqDAwpwAK3NEREREREQixMocERERERGJAsfMCTGZIyIiIiIiUWAuJ8RkjojkJimdjyYoKnzbV5Z3CPReDp8YVGRwOvyiIz0zR94hEBVJTOaIiIiIiEgUJGBp7lOcAIWIiIiIiEiEWJkjIiIiIiJR4KMJhFiZIyIiIiIiEiFW5oiIiIiISBT4aAIhVuaIiIiIiIhEiJU5IiIiIiISBRbmhJjMERERERGRKCgwmxNgN0siIiIiIiIRYmWOiIiIiIhEgYU5IVbmiIiIiIiIRIiVOSIiIiIiEgU+mkCIlTkiIiIiIiIRYmWOiIiIiIhEgYU5IVbmiIiIiIiIRIiVOSIiIiIiEgU+Z06IyRwREREREYkCUzkhdrMkIiIiIiISIVbmiIiIiIhIFPhoAiFW5oiIiIiIiESIlTkiIiIiIhIFBRbmBFiZIyIiIiIiEiFW5oiIiIiISBQ4Zk6IlTkiIiIiIiIRYmWOiIiIiIhEgYU5ISZzREREREQkCuxmKcRulkRERERERCLEyhwREREREYkCH00gxMocERERERGRCLEyR0REREREosAxc0KszBEREREREYkQK3NERERERCQKrMsJsTJHREREREQkQqzMERERERGRKChwzJwAK3NEREREREQiVODKXIcOHQp80F27dn1TMERERERERPlhYU6owMmcrq5uYcZBRERERET0WXw0gVCBk7mAgIDCjIOIiIiIiIi+AidAISIiIiIiUWBhTuibk7kdO3Zg+/btCA8PR0ZGhmDb1atX/+/AiIiIiIiIKH/flMz5+flhypQpcHd3x969e9GnTx88evQIly5dwrBhw753jETF3vPnzzFjxgwcPnwYr1+/hpmZGdq1a4fp06fD0NCwQMd4+vQprKysEBoaCkdHx8INWORuXruCHVs24sHdMMTFvsJ076Wo2+AXAEBWViYC167ApZCziIx4AU1NbVStWRt9B4+EoXEJ2THeJiZg1ZL5uBB8ChIFBTg3aoIhIydAXUNDXpf1U/itza+IiozItb5dp64YPWEqRg5yx7WrlwXb2nTojLGTZvyoEIu15OQkrFruh3+PH8ObuFhUqGiL8ROnoJK9vbxD+6ncCL2Mv/7ciAf3whD7+hVmLfBFvYa/yLZLpVJsXLcKB/fuRFLSW1S2d8TI8VNRqnQZWZvEhASs8PFGyNl3n1H1G7vAYzQ/o77WzWtXsHNrIB7ee/fzYuq8JbKfFwDw5+/+OH38CF7FREFZSRnWFezQe4AHKlb6+H9i1sSRePzgHuLj46ClpQPHGrXRd8hIGBqVyOuUxR4fTSD0TY8mWLVqFdauXYvly5dDRUUF48ePR1BQEEaMGIGEhITvHSNRsfb48WPUqFEDDx48wNatW/Hw4UOsXr0ax48fh5OTE+Li4r7r+f5baS+O0lJTYWVdAcPGTsq1LT0tDQ/v3UV394FY8ftfmOa1BC/Cn2LmhJGCdgtmTcKzJ4/g5bsasxb64da1q1i2cPaPuoSf1prAbdh16KRs8VmxDgDQyOVXWZtW7ToJ2gwePlZe4RY7s6dPw/mQc5jrvQDbd++DU11nDB7QBzHR0fIO7aeSmpqKcjYVMGLc5Dy3b/sjALu3b8GoCdOwYv1mqKmrY+KowchIT5e18ZoxEU+fPMJCvzWYt3g5boZewZL5s37UJfw00tJSYWVdHkPH5P55AQAlLcpgyOiJWBW4A4tWBaCEqTmmjh2ChDcff3ZXqVoDk2YvxNrNezBl7mJERTyH17RxP+oSSOS+KZkLDw9H3bp1AQDq6up4+/YtAKBXr17YunXr94uOiDBs2DCoqKjg6NGjaNiwIUqXLo3mzZvj2LFjePnyJaZMmQLg3exOe/bsEeyrp6eHjRs3AgCsrKwAAFWrVoVEIkGjRo0AAO7u7mjXrh3mzZsHc3NzVKhQAQBw8+ZN/PLLL1BXV4ehoSEGDhyIpKQk2bFPnjyJWrVqQVNTE3p6enB2dsazZ88K92b8IDWd6sF9oAecGzbJtU1TSxvey9agQRNXWJSxhG3lKhg6ZhIe3LuDmKhIAED408e4fD4YoybOQMVKVVDZoRqGjp6IU8cOI/ZVzI++nJ+Knr4BDI2MZEvI2VMoWcoCjtVqytqoqakJ2mhqackx4uIjLS0Nx48dxagx41C9Rk2ULl0Gg4cNh0Xp0vj7L/5u8D3VrlsffQcPR71GuT+jpFIpdv31J3r2GQDnBo1RzqY8JsyYh9evX+Hs6X8BAM+ePMal88EYO3kmbCtXgb1jNXiMnYgTQYfxmp9RX6VmnXpwG+AhqMZ9qnHTFqhaow7MzEuhjJU1Bg4fi5TkJDx59EDWpv1vvVCxUhWYmJrDzt4RnXv0xd3bN5GVlfmjLkNUJJLCW8Tom5I5U1NTWTWgdOnSOH/+PADgyZMnkEql3y86omIuLi4OR44cwdChQ6Guri7YZmpqih49euCvv/4q0P+7ixcvAgCOHTuGyMhIwfMgjx8/jnv37iEoKAj79+9HcnIyXF1doa+vj0uXLuHvv//GsWPH4OHhAQDIyspCu3bt0LBhQ9y4cQMhISEYOHBgsZ0uODkpCRKJBJra2gCAsFvXoaWtjfK2lWRtqtaoDYmCAu7euSmvMH86mZmZCDq0H83btBf82ws6fABtXOrB/bd2WLtiKdLSUuUYZfGRnZ2F7OxsqKiqCtarqqoh9OoVOUVV/ERGvERc7GtUq1lHtk5LSxu2lexx5+Z1AMCd959RFT75jKpes867z6jb/IwqLJmZmTi0byc0tbRgZV0+zzZvExNwIuggbCs7QElJ+QdHSGL0TWPmfvnlF+zbtw9Vq1ZFnz59MHr0aOzYsQOXL1/+qoeLE9HnPXjwAFKpFLa2tnlut7W1xZs3b/Dq1asvHsvY2BgAYGhoCFNTU8E2TU1NrF+/HioqKgCAdevWIS0tDZs2bYKmpiYAYMWKFWjdujUWLFgAZWVlJCQkoFWrVihXrpwsls9JT09H+iddfN6tk0L1P7/4iU1Gejp+9/dFI5fm0NR8VwF6ExsLXT0DQTtFJSVoa+vgTVysPML8KZ05eRxJSW/RvFU72bomri1hamYOQ2NjPH5wH2tWLEX4s6eYu2iZ/AItJjQ1tVDFwRHrVq+CVdmyMDQ0wuGDB3Dj+jVYlC4t7/CKjTexrwEA+gbC8dT6BoZ4E/vu8ycu9jX09HN/Runo6CDu/f70/VwIPo0FsyYgPS0NBoZGmLdkNXT19AVtfvf3xT+7tiE9LQ0VK1XBzAV+coq26CuuXxzn55sqc2vXrpV17Ro2bBh+//132NraYvbs2fD39/+uARIRCr3ibW9vL0vkACAsLAwODg6yRA4AnJ2dkZOTg3v37sHAwADu7u5wdXVF69atsWzZMkRGRn72HN7e3tDV1RUs/ssWFdo1/QhZWZmYN80TUqkUHp5T5B1OsXNw3y7UcqoHo08mnmnToTNqOTmjnHV5NG3eCpNneuHMyeN4+SJcjpEWH3O9F0IKKVx/aYja1apg6+Y/0Kx5SyhIvunXDaKfgkO1mljx+1/w8Q9E9drO8J4xHvFvhOPdO3Zzw/INf2HuEn8oKCjAZ+5U9nbLh0IhLmL0TXErKChASeljUa9r167w8/PD8OHDBb8QEtH/x9raGhKJBGFhYXluDwsLg76+PoyNjSGRSHJ98GdmFqy//adJW0EFBAQgJCQEdevWxV9//YXy5cvLulznZdKkSUhISBAsQ0Z6fvV5i4qsrEx4TfNETHQkvH3XyKpyAKBvaIiEeOEP6uysLLx9m5jr23L6NlGREbhy8Txatev42Xa2ld/NGPfy+fMfEVaxZ1G6NDZs/BPnLl7FoWMn8Oe2v5GVlYWSpSzkHVqxoW9oBAC5egG8iYuF/vvZjw0MjXIlE9lZWUhMTITB+/3p+1FTV4d5qdKoWKkKRk2cCUVFRRzZv1vQRldPH6VKl0G1mk6YOHMBLp0/i7u3b8gpYhKTb05Cz5w5g549e8LJyQkvX74EAPzxxx84e/bsdwuOqLgzNDRE06ZNsWrVKqSmCsf9REVFYfPmzfjtt98gkUhgbGwsqI49ePAAKSkpstcfvmjJzs7+4nltbW1x/fp1JCcny9YFBwdDQUFBNkEK8G4ylUmTJuHcuXOoXLkytmzZku8xVVVVoaOjI1jE2sXyQyL38nk4vH3XQEdXT7DdtrIDkt6+xYO7d2Trrl25CGlODiracYr27+HQP7uhp2+AOs4NPtvu4f27AABDI/6C+iOpa2jA2LgEEhMScO7cWTT6Je/JIej7MzMvCQNDI1y9dEG2Ljk5CWG3b8LO3gEAYPf+M+r+J59RoR8+oyrxM6qw5eRIkZmZ/8zROdIcAPhsm+JMIpEU2iJG35TM7dy5E66urlBXV0doaKhsHExCQgK8vLy+a4BExd2KFSuQnp4OV1dXnD59Gs+fP8fhw4fRtGlTlCxZEvPmzQPwbizrihUrEBoaisuXL2Pw4MFQVv44eLpEiRJQV1fH4cOHER0d/dnHiPTo0QNqampwc3PDrVu3cOLECQwfPhy9evWCiYkJnjx5gkmTJiEkJATPnj3D0aNH8eDBgy+OmxOL1JQUPLp/F4/eJwJRES/x6P5dxERFIisrE3OnjMP9u3cwYYY3cnJyEBf7GnGxr2WV0NKWZVGjjjN8F8zCvTs3cftGKFYt9UZDl2aCZ9HRt8nJycGhf/agWcu2gl4iL1+EI3D9atwLu43IiJcIPnUCXjMmw6FqDZSzqfCZI9L3ci74DILPnsHLFy9w/lwwBvR1g5VVWbRpx/H031NqSgoe3r8r+7IiKuIlHt6/i+ioSEgkEnT4rSc2b1yLc6dP4PHD+5g/awqMjIxR7/2Mi2WsyqJmHWf4eM3E3ds3cet6KPwWe6Nx02aCbsv0ZakpKXj04C4ePXj3XkRHvsSjB3cREx2JtNRUbFzjh7u3byA6KgIP7t3BUu8ZiH0dg/qNmwIA7t6+iX92bsOjB3cRHRWBa1cuYsHMiTAraQHbSg7yvDQSCYn0GzrkVq1aFaNHj0bv3r2hra2N69evo2zZsggNDUXz5s0RFRVVGLESFVvPnj2TPTQ8Li4OpqamaNeuHWbMmCF7aHhERAT69OmD4OBgmJubY9myZejWrRt8fX3h7u4OAFi/fj1mz56Nly9fon79+jh58iTc3d0RHx+f67EGN2/exMiRIxESEgINDQ107NgRS5YsgZaWFqKjozF48GBcuHABsbGxMDMzg5ubG2bMmAEFhYJ/R/Tkddr3ukXf1fWrlzBheP9c612at0HPfoPh3qlFnvstWL4eDu+nyH+bmICVS7xx4f0Dees1aoIhoyYW2QfyqqsoyjuEArt0Phjjhg/Cnzv2w6KMpWx9TFQk5k6fhCePHyAtNRXGJqao36gJevcdJKrHE+iof9PcZEXC0cOHsNx3CaKjo6Crq4cmTZti2IjR0H4/06vYxCUVzcrItSuXMHZYv1zrf23RBhOmz5U9NPzAnh1ISnoL+ypVMWL8FFiUtpS1TUxIwHIfL4ScPQUFyfuHho8pup9R6Zk58g4hTzdCL2HiiAG51rs0aw2PcVOxcPYk3LtzEwkJ8dDR0UN520ro2rs/yttWBgA8efQAa/wW4snD+0hLS4WBoRGq13JGV7f+MDI2+dGXU2DlSqh/uVEhGbX3bqEd27dtxUI7dmH5pmROQ0MDd+7cgaWlpSCZe/z4Mezs7JCWVjR/QSOioqWoJnPFkZiSuZ+dmJO5n01RTeaKo6KazBVXTOaKjm9+ztzDhw9zrT979izKli37fwdFRERERET0XwqSwlvE6JuSuQEDBmDkyJG4cOECJBIJIiIisHnzZowdOxZDhgz53jESERERERHRf3xTX46JEyciJycHTZo0QUpKCho0aABVVVV4enqif//c40yIiIiIiIj+X2KddbKwfFNlTiKRYMqUKYiLi8OtW7dw/vx5vHr1Crq6urCysvreMRIREREREbGb5X98VTKXnp6OSZMmoUaNGnB2dsbBgwdhZ2eH27dvo0KFCli2bBlGjx5dWLESERERERHRe1/VzXL69OlYs2YNXFxccO7cOXTu3Bl9+vTB+fPn4ePjg86dO0NRkTOiERERERHR98delkJflcz9/fff2LRpE9q0aYNbt26hSpUqyMrKwvXr19l/lYiIiIiI6Af6qmTuxYsXqF69OgCgcuXKUFVVxejRo5nIERERERFRoVNg3iHwVWPmsrOzoaKiInutpKQELS2t7x4UERERERERfd5XVeakUinc3d2hqqoKAEhLS8PgwYOhqakpaLdr167vFyERERERERG+cSr+n9hXJXNubm6C1z179vyuwRAREREREVHBfFUyFxAQUFhxEBERERERfRaHzAl9VTJHREREREQkL5wARYjdTomIiIiIiESIlTkiIiIiIhIFFuaEWJkjIiIiIiISIVbmiIiIiIhIFBRYmRNgZY6IiIiIiEiEWJkjIiIiIiJR4GyWQqzMERERERERiRArc0REREREJAoszAkxmSMiIiIiIlHgBChC7GZJREREREQkQqzMERERERGRKEjA0tynWJkjIiIiIiISIVbmiIiIiIhIFDhmToiVOSIiIiIiIhFiZY6IiIiIiESBlTkhVuaIiIiIiIi+0fz58yGRSDBq1CjZurS0NAwbNgyGhobQ0tJCx44dER0dLdgvPDwcLVu2hIaGBkqUKAFPT09kZWV91bmZzBERERERkShIJJJCW77FpUuXsGbNGlSpUkWwfvTo0fjnn3/w999/49SpU4iIiECHDh1k27Ozs9GyZUtkZGTg3LlzCAwMxMaNGzF9+vSvOj+TOSIiIiIiEgUFSeEt6enpSExMFCzp6en5xpKUlIQePXpg3bp10NfXl61PSEjAhg0bsGTJEvzyyy+oXr06AgICcO7cOZw/fx4AcPToUdy5cwd//vknHB0d0bx5c8yZMwcrV65ERkZGwe/Ht99KIiIiIiKin4O3tzd0dXUFi7e3d77thw0bhpYtW8LFxUWw/sqVK8jMzBSsr1ixIkqXLo2QkBAAQEhICOzt7WFiYiJr4+rqisTERNy+fbvAMXMCFCIiIiIiEoVv7A1ZIJMmTcKYMWME61RVVfNsu23bNly9ehWXLl3KtS0qKgoqKirQ09MTrDcxMUFUVJSszaeJ3IftH7YVFJM5IiIiIiIq9lRVVfNN3j71/PlzjBw5EkFBQVBTU/sBkeWP3SyJiIiIiEgUFCSSQlsK6sqVK4iJiUG1atWgpKQEJSUlnDp1Cn5+flBSUoKJiQkyMjIQHx8v2C86OhqmpqYAAFNT01yzW354/aFNge5HgVsSEREREREVc02aNMHNmzdx7do12VKjRg306NFD9ndlZWUcP35cts+9e/cQHh4OJycnAICTkxNu3ryJmJgYWZugoCDo6OjAzs6uwLGwmyUREREREYlCUXhouLa2NipXrixYp6mpCUNDQ9n6fv36YcyYMTAwMICOjg6GDx8OJycn1KlTBwDw66+/ws7ODr169cLChQsRFRWFqVOnYtiwYQXq6vkBkzkiIiIiIqLvaOnSpVBQUEDHjh2Rnp4OV1dXrFq1SrZdUVER+/fvx5AhQ+Dk5ARNTU24ublh9uzZX3UeiVQqlX7v4ImICuLJ6zR5h0DvqasoyjsEek9Hnd+zFhVxSQV/1hMVrvTMHHmHQJ8oV0JdbudeHvyk0I493Nmq0I5dWPgTg4iIiIiIREEBRaCfZRHCCVCIiIiIiIhEiJU5IpIbfU1leYdA76kqsZtlUVGYD8Slr1Nr0gF5h0DvHZ/uKu8QqIjgZ6QQK3NEREREREQixMocERERERGJQlF4NEFRwsocERERERGRCLEyR0REREREoqDAQXMCrMwRERERERGJECtzREREREQkCizMCTGZIyIiIiIiUWA3SyF2syQiIiIiIhIhVuaIiIiIiEgUWJgTYmWOiIiIiIhIhFiZIyIiIiIiUWAlSoj3g4iIiIiISIRYmSMiIiIiIlGQcNCcACtzREREREREIsTKHBERERERiQLrckKszBEREREREYkQK3NERERERCQKChwzJ8BkjoiIiIiIRIGpnBC7WRIREREREYkQK3NERERERCQK7GUpxMocERERERGRCLEyR0REREREosCHhguxMkdERERERCRCrMwREREREZEosBIlxPtBREREREQkQqzMERERERGRKHDMnBCTOSIiIiIiEgWmckLsZklERERERCRCrMwREREREZEosJulECtzREREREREIsTKHBERERERiQIrUUK8H0RERERERCLEyhwREREREYkCx8wJsTJHREREREQkQqzMERERERGRKLAuJ8RkjoiIiIiIRIG9LIXYzZKIiIiIiEiEWJkjIiIiIiJRUGBHSwFW5uincOzYMaxfv17eYRARERER/TBM5kRm5syZcHR0lHcY3+zkyZOQSCSIj4//bse8f/8+3N3dUatWrQK1l0gk2LNnz3c7P1A41wXkfr/d3d3Rrl2773oOIiIiIrGQSApvESN2s/xO3N3dERgYCABQUlJCqVKl0LlzZ8yePRtqampyjq7g+vTpg5IlS2Lu3Lm5tsnrGhs1agRHR0f4+vrm2paamoru3bsjICAAVapUKdDxIiMjoa+v/52j/DGWLVsGqVT6Q88pkUiwe/duJpHvBf6+Dqv8luK37r0wZvwkAEB6ejqW+SxE0JGDyMzIQO269TB+8jQYGhrJOdqfn//K5Vjjv0KwztLKCnv+OSyniGjbls0IDNiA169foXyFipg4eRrsC/j5TAVzYZ4rLAw1c63f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CE1NTYSEhKBBgwYAypZZXr58GZMmTXqjr4mISNVwZo6IiIjKGTZsGGrWrIm+ffsiODgYCQkJOHPmDLy9vXH//n2l68LDw/Hdd9/h008/hZaW1ksfLz8/HxMmTMCZM2dw7949hISE4PLly7C1tQUA2NjY4MSJEwgNDUVUVBS+/PJLPH78WPHxenp6GDduHKZNm4ajR4/i9u3b8PLyQl5eHr744ot3+80gInpPcWaOiIiIytHV1cW5c+cwffp0eHh4IDs7G+bm5nB1dVWaqbO2toaTkxPCwsLg7+//ysdTV1dHWloaRowYgcePH6NmzZrw8PDAggULAACzZ89GfHw83NzcoKurizFjxqBfv37IzMxUPMbSpUtRWlqKzz77DNnZ2WjdujWOHTsGExOTd/Z9ICJ6n7GbJRERERERURXEZZZERERERERVEIs5IiIiIiKiKojFHBERERERURXEYo6IiIiIiKgKYjFHRERERERUBbGYIyIiIiIiqoJYzBEREREREVVBLOaIiIiIiIiqIBZzREREREREVRCLOSIiIiIioiqIxRwREREREVEV9P8A+Cl+wIpnMSEAAAAASUVORK5CYII=",
            "text/plain": [
              "<Figure size 800x600 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=clf_tradicional.classes_, yticklabels=clf_tradicional.classes_)\n",
        "plt.xlabel('Previsão')\n",
        "plt.ylabel('Real')\n",
        "plt.title('Matriz de Confusão')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-J86UCvOb5Ug"
      },
      "source": [
        "Possui o objetico de criar e exibir um mapa de calor (heatmap) visualizando a matriz de confusão calculada anteriormente."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ff73n0Pqb5d0"
      },
      "source": [
        "### **Visualização - Curva ROC**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "t8iqoisOb5pf"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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",
            "text/plain": [
              "<Figure size 800x600 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "y_probas = clf_tradicional.predict_proba(test_vectors_tradicional)\n",
        "skplt.metrics.plot_roc(test_labels_tradicional, y_probas, figsize=(8, 6), plot_micro=False)\n",
        "plt.title('Curva ROC Tradicional')\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "WCQHA6aJdRAK"
      },
      "source": [
        "Com o objetivo de plotar a Curva ROC (Receiver Operating Characteristic) para avaliar o desempenho do modelo de classificação Naive Bayes Multinomial tradicional nos dados de teste."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HxQKkfmGPEn8"
      },
      "source": [
        "# **Parte 2 – Realizar a tarefa de classificação apresentada no item anterior com a utilização IA Generativa.**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iw27Q70_gWzO"
      },
      "source": [
        "# Carregar os dados"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "ZkLdVIydvLv7"
      },
      "outputs": [],
      "source": [
        "url = \"https://raw.githubusercontent.com/thiagonogueira/datasets/main/tickets_reclamacoes_classificados_one_line_generative.csv\"\n",
        "df = pd.read_csv(url, delimiter=';')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 363
        },
        "id": "m3iZo7-D7AZt",
        "outputId": "42f0f4ea-78fb-4ee6-e3ac-d785561e1db9"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
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
              "      <th>id_reclamacao</th>\n",
              "      <th>data_abertura</th>\n",
              "      <th>categoria</th>\n",
              "      <th>descricao_reclamacao</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>3409650</td>\n",
              "      <td>2019-10-18T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>Chase afirma que eles me enviaram uma carta em...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2824212</td>\n",
              "      <td>2018-02-23T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>Em xx/xx/xxxx, tentei usar meu chase xxxx, ele...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3322111</td>\n",
              "      <td>2019-07-29T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>JPMCB - Inquérito de serviço do cartão do cart...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3435102</td>\n",
              "      <td>2019-11-11T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>O Chase Bank me relatou as agências de crédito...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>3556741</td>\n",
              "      <td>2020-03-06T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>Começando o XX/XX/2016, fui vítima de empresas...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>3637331</td>\n",
              "      <td>2020-05-05T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>Esta reclamação está no Ref à reclamação inici...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>2682665</td>\n",
              "      <td>2017-09-23T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>Meu nome está nesses empréstimos apenas porque...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>3378267</td>\n",
              "      <td>2019-09-18T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>Meu perfil XXXX mostra duas perguntas recentes...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>3676350</td>\n",
              "      <td>2020-05-31T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>Em ou sobre xx/xx/xxxx, perguntei para alugar ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>1463619</td>\n",
              "      <td>2015-07-11T12:00:00-05:00</td>\n",
              "      <td>Cartão de crédito / Cartão pré-pago</td>\n",
              "      <td>Recentemente, tive uma taxa de cartão de crédi...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   id_reclamacao              data_abertura  \\\n",
              "0        3409650  2019-10-18T12:00:00-05:00   \n",
              "1        2824212  2018-02-23T12:00:00-05:00   \n",
              "2        3322111  2019-07-29T12:00:00-05:00   \n",
              "3        3435102  2019-11-11T12:00:00-05:00   \n",
              "4        3556741  2020-03-06T12:00:00-05:00   \n",
              "5        3637331  2020-05-05T12:00:00-05:00   \n",
              "6        2682665  2017-09-23T12:00:00-05:00   \n",
              "7        3378267  2019-09-18T12:00:00-05:00   \n",
              "8        3676350  2020-05-31T12:00:00-05:00   \n",
              "9        1463619  2015-07-11T12:00:00-05:00   \n",
              "\n",
              "                             categoria  \\\n",
              "0  Cartão de crédito / Cartão pré-pago   \n",
              "1  Cartão de crédito / Cartão pré-pago   \n",
              "2  Cartão de crédito / Cartão pré-pago   \n",
              "3  Cartão de crédito / Cartão pré-pago   \n",
              "4  Cartão de crédito / Cartão pré-pago   \n",
              "5  Cartão de crédito / Cartão pré-pago   \n",
              "6  Cartão de crédito / Cartão pré-pago   \n",
              "7  Cartão de crédito / Cartão pré-pago   \n",
              "8  Cartão de crédito / Cartão pré-pago   \n",
              "9  Cartão de crédito / Cartão pré-pago   \n",
              "\n",
              "                                descricao_reclamacao  \n",
              "0  Chase afirma que eles me enviaram uma carta em...  \n",
              "1  Em xx/xx/xxxx, tentei usar meu chase xxxx, ele...  \n",
              "2  JPMCB - Inquérito de serviço do cartão do cart...  \n",
              "3  O Chase Bank me relatou as agências de crédito...  \n",
              "4  Começando o XX/XX/2016, fui vítima de empresas...  \n",
              "5  Esta reclamação está no Ref à reclamação inici...  \n",
              "6  Meu nome está nesses empréstimos apenas porque...  \n",
              "7  Meu perfil XXXX mostra duas perguntas recentes...  \n",
              "8  Em ou sobre xx/xx/xxxx, perguntei para alugar ...  \n",
              "9  Recentemente, tive uma taxa de cartão de crédi...  "
            ]
          },
          "execution_count": 13,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "df.head(10)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6XZ-Y9t-va0f",
        "outputId": "a5755736-c2e1-43fe-d2c3-79e6462491a7"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Nomes das Colunas: Index(['id_reclamacao', 'data_abertura', 'categoria', 'descricao_reclamacao'], dtype='object')\n"
          ]
        }
      ],
      "source": [
        "print(\"Nomes das Colunas:\", df.columns)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "eXPptlB2vsvi"
      },
      "outputs": [],
      "source": [
        "column_tradicional = df.columns[3]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wBeWhQcTgWzP"
      },
      "source": [
        "# Dividir os dados em treinamento e teste"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "euU3ivNQvV3g"
      },
      "outputs": [],
      "source": [
        "train_data, test_data, train_labels, test_labels = train_test_split(\n",
        "    df[column_tradicional], df['categoria'], test_size=0.2, random_state=42\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "TM9uUjXbgWzP"
      },
      "source": [
        "# Criar um modelo TF-IDF"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "D73BEbBmzhe1"
      },
      "outputs": [],
      "source": [
        "tfidf_vectorizer = TfidfVectorizer(max_features=5000)\n",
        "tfidf_train = tfidf_vectorizer.fit_transform(train_data)\n",
        "tfidf_test = tfidf_vectorizer.transform(test_data)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xJAb-dwagWzP"
      },
      "source": [
        "# Treinar um classificador Naive Bayes"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 75
        },
        "id": "0sCWD_X3zlxT",
        "outputId": "09210cd3-7242-4034-d88d-68218c6e9cf5"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MultinomialNB</label><div class=\"sk-toggleable__content\"><pre>MultinomialNB()</pre></div></div></div></div></div>"
            ],
            "text/plain": [
              "MultinomialNB()"
            ]
          },
          "execution_count": 18,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "nb_classifier = MultinomialNB()\n",
        "nb_classifier.fit(tfidf_train, train_labels)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LuBlR3htgWzP"
      },
      "source": [
        "# Fazer previsões no conjunto de teste"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "XZhCMPiv1ndP"
      },
      "outputs": [],
      "source": [
        "predictions = nb_classifier.predict(tfidf_test)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Hfmw8STNgWzQ"
      },
      "source": [
        "# Avaliar o desempenho do modelo"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "aamwYwiJ1osn"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Acurácia: 0.805\n",
            "\n",
            "Relatório de Classificação:\n",
            "                                      precision    recall  f1-score   support\n",
            "\n",
            "Cartão de crédito / Cartão pré-pago       0.83      0.76      0.79        33\n",
            "            Hipotecas / Empréstimos       1.00      0.67      0.80        48\n",
            "                             Outros       0.69      0.89      0.78        37\n",
            "       Roubo / Relatório de disputa       0.70      0.92      0.80        38\n",
            "         Serviços de conta bancária       0.90      0.82      0.86        44\n",
            "\n",
            "                           accuracy                           0.81       200\n",
            "                          macro avg       0.82      0.81      0.80       200\n",
            "                       weighted avg       0.84      0.81      0.81       200\n",
            "\n"
          ]
        }
      ],
      "source": [
        "accuracy = accuracy_score(test_labels, predictions)\n",
        "print(f'Acurácia: {accuracy}')\n",
        "print('\\nRelatório de Classificação:\\n', classification_report(test_labels, predictions))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "izApqPOuiU-i"
      },
      "source": [
        "# Integração com GPT-3"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 35,
      "metadata": {
        "id": "72i8tZ63DJ5z"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "openai.api_key = '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'"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dq4h52UsgWzQ"
      },
      "source": [
        "# Função para gerar texto usando GPT-3"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 36,
      "metadata": {
        "id": "omOvUNGSDSaV"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "def generate_text(prompt):\n",
        "    response = openai.Completion.create(\n",
        "      engine=\"text-davinci-003\",\n",
        "      prompt=prompt,\n",
        "      max_tokens=10\n",
        "    )\n",
        "    return response['choices'][0]['text']"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xYPPIJ5lifsH"
      },
      "source": [
        "Busca no CSV https://raw.githubusercontent.com/thiagonogueira/datasets/main/tickets_reclamacoes_classificados_one_line_generative.csv"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 37,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "RXy1yIopiguh",
        "outputId": "02f9bbd9-7c03-4406-a0ed-1cae68dca2f1"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Texto: Lamento não ter documentado as datas em que entrei em contato com o departamento xxxx da Chase Bailruptcy. Foi algumas vezes desde o ano xx/xx/2011. Perguntei a eles por que eles não relatariam meus pagamentos (que eram todos oportunos) ao Credit Bureau e me disseram porque fui falido. Não entendo, pois pedi ao meu advogado que não os incluísse na falência.   Na minha última conversa com eles, perguntei a eles; \"Se após o período de 10 anos, que em breve, você relatará meus pagamentos então?   Eles disseram que não. Em outras palavras, esqueça isso.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "Isso é muito decepc\n",
            "======================================================\n",
            "Texto: Essa ofensa específica começou em xx/xx/xxxx e agora em xx/xx/xxxx. Meu ACOOUNT sempre esteve em todos os alertas, de alguma forma esses alertas mantinham/continuam sendo 'redefinidos'. Em xx/xx/xxxx, um banqueiro telefônico redefine meus alertas para ter um período de blecaute que não funciona para mim. Fui acusado de outro, de muitos, taxas de fundos tardios/cheios de cheque especial/insuficiente por causa de seus problemas ou situações do sistema. Eu tive minha conta bancária com Chase desde xx/xx/xxxx/xx/xx/xxxx e nunca tive tantos problemas ultimamente. Estou no Seguro Social e XXXX, então tenho que executar minha conta corrente da maneira que vejo em forma e, durante anos, funcionou até recentemente. O Chase possui um sistema de computador que continua mudando os alertas das pessoas e eles começam a cobrar todas as taxas altas do final. Eu gostaria de reembolsar minha taxa. Chase não é um bom banco para pessoas que não são ricas, estou procurando uma nova instituição financeira. Minhas configurações estavam preparadas para me avisar se eu estava entrando em cheque especial e me permitiu cobrir esse cheque especial com um rápido depósito antes do XXXX Eastern Hora. Com o computador, o computador não está me enviando os alertas que eu configurei ou solicitou. Isso é falso e parece ser uma prática comercial ilegal para mim.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Informei o Chase no XX/XX/XXXX através do site deles que eu queria iniciar as disputas abaixo, pois esses itens foram cancelados e não consegui que o comerciante respondesse. Quero que Chase contestasse essas cobranças em meu nome e entre em contato com o comerciante para me reembolsar.   Data da postagem Descrição Categoria Tipo Valor xx/xx/2020 XXX Compras Venda xxxx xx/xx/2020 XXXX COMPRAS VENDA XXXX XX/XX/2020 XXXX VENDA XXXX XX/XX/2020 XXXX COMPRAS VENDA XXXX XX/XX/2020 XXXX COMPLAR XXXX XX/XX/2020 XXXX COMPRAS VENDA XXXX\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "Informei o Chase no XX/\n",
            "======================================================\n",
            "Texto: Entre xx/xx/xxxx e xx/xx/xxxx, meu cartão de débito foi clonado ilegalmente como usado para várias transações fraudulentas, totalizando mais de {$ 800,00}. A detecção de fraude e o banco on -line do JP Morgan Chase não conseguiu pegar a fraude. Quando finalmente telefonei para cancelar o cartão de débito e sinalizar as transações, me disseram que não seria reembolsado o valor total roubado ilegalmente da minha conta, devido a um limiar de fraude de 60 dias que o banco tem em reembolso, o dinheiro roubado. Eu quero ser feito inteiro.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, enviei meu pagamento de hipoteca xx/xx/xxxx de {$ 1200.00} para perseguir por correio certificado. Chase assinou o pagamento em xx/xx/xxxx. Chase aplicou o pagamento da hipoteca como um pagamento de redução principal {$ 1200,00}. Em xx/xx/xxxx, o Chase reverteu o pagamento de redução principal {$ 1200.00} e reaplicou o pagamento da hipoteca xx/xx/xxxx como um pagamento de redução principal de {$ 1200,00}. Entrei em contato com o Chase diariamente para corrigir o pagamento da hipoteca xx/xx/xxxx. A partir de xx/xx/xxxx, o Chase não corrigiu nosso pagamento de hipoteca xx/xx/xxxx. O Chase não nos enviou nossa instrução hipotecária xx/xx/xxxx ou xx/xx/xxxx, apesar de ter contatado repetidamente para nossas instruções de hipoteca.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Portanto, estamos em processo de\n",
            "======================================================\n",
            "Texto: Olá, estou procurando registrar uma reclamação contra o Chase Bank em relação às políticas e procedimentos desfavoráveis ​​projetados para aproveitar os clientes.   Como pano de fundo, fiz reservas de voos de companhias aéreas usando o site da Chase Travel usando seu cartão de crédito premium (Chase Sapphire Reserve com {$ 550,00} taxa anual). A companhia aérea cancelou o voo devido à situação do Covid-19. De acordo com as políticas da companhia aérea, sou elegível para um reembolso total. No entanto, como a reserva foi feita através da Chase Travel (agente), o reembolso deve ser iniciado através delas.   Conversei com a busca da equipe de viagem 7 vezes coletivamente por 7-8 horas e tudo o que eles fazem é enviar um email para a companhia aérea para fornecer um código para processar o reembolso. Isso acontece há 4 meses e me disseram repetidamente para continuar esperando até que eles recebam uma resposta ao endereço de e -mail. Para piorar as coisas, insistindo em investigar o assunto, eles descobriram que não têm endereço de e -mail correto (conforme confirmado por eles no email). No entanto, isso ainda não resolveu nada. Eu tenho 2 casos abertos com perseguir por 3 meses.   As políticas e procedimentos da Chase Travel são projetados para prejudicar os clientes e os representantes de atendimento ao cliente não recebem treinamento sobre como resolver/escalar esses problemas para o próximo nível. Além disso, nenhum monitoramento é feito nas disputas/reclamações do cliente e os 2 casos acima ainda estão abertos por meses.   Peço ao CFPB que tome medidas contra o Chase Bank e faça com que eles resolvam esse problema. Além disso, como explicado acima, ele pode exigir uma revisão e atualização mais amplas para as políticas e procedimentos de viagem Chase para torná -lo justo para os clientes.   Detalhes do vôo: perseguir viagem - Número do itinerário: XXXX Data de viagem: xxxx xxxx para xx/xx/2020 reembolso solicitado: xxxx\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  xxxx.\n",
            "A CFPB pede\n",
            "======================================================\n",
            "Texto: Encontrei cobranças fraudulentas no meu cartão de crédito.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "Primeiro, é importante notific\n",
            "======================================================\n",
            "Texto: Eu tive um empréstimo com o Chase Auto e sofri em um acidente de carro, o carro foi totalizado e minha companhia de seguros XXXX me disse para não fazer os pagamentos porque eles pagariam o carro, mas levaram três meses para fazer isso, mas fiquei Seguindo com XXXX e Chase, mas de alguma forma como eles estão relatando no meu relatório de crédito\n",
            "Predição Tradicional: Outros\n",
            "Resposta: , estou tendo pontuação de\n",
            "======================================================\n",
            "Texto: O Banco não permitirá que o acesso ou o aplicativo não tenha acesso ao acesso para pedir dinheiro emprestado\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: . Além disso, será necess\n",
            "======================================================\n",
            "Texto: Eu contestei uma cobrança de {$ 2000,00} desde a data em que foi cobrada (xx/xx/xxxx). De acordo com a Lei de Relatórios de Crédito Justo, o fornecimento de informações às agências de relatórios de consumidores tem o dever de fornecer informações precisas sobre um arquivo de crédito de consumidores. 15 U.S.C. 1681S-2 (a). Como nego a responsabilidade por qualquer quantia de suposta dívida e, como afirmo que a suposta dívida resultou de uma acusação não autorizada, não é preciso relatar que há um pagamento devido ao pagamento devido. Apesar de uma carta (anexada) do meu advogado para lembrar a Chase My Fair Credit Reporting Right, Chase continua a violar a lei. Apesar do meu esforço repetido para resolver essa disputa, pedindo arbitragem em xx/xx/xxxx por nosso contrato de cartão, o Chase ignora a solicitação de arbitragem (um XXXX compatível com CFPB separado foi arquivado há 2 semanas). Corrija os relatórios de crédito a todas as agências de crédito o mais rápido possível, pois minha pontuação de crédito sofreu XXXX pontos diminuíram.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Se a cobrança não for\n",
            "======================================================\n",
            "Texto: Um cheque de 1000 dólares foi depositado na minha conta corrente. Eu tinha chamado Chase para cancelar o cheque e não processar o cheque, mas não consegui fazer nada a respeito e resultou no pagamento do dobro do valor.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "\n",
            "É importante entrar em contato\n",
            "======================================================\n",
            "Texto: Investigue as práticas de relatórios de pontuação de crédito bancário Chase. Eles estão discriminando com base na composição ou renda do bairro xxxx quando distribuem \"as pontuações de crédito gratuitas\". Evidências de hoje xx/xx/18? Minha pontuação FICA com xxxx é xxxx, com xxxx pontuação de crédito é xxxx, ambas excelentes classificações e merecidamente. Minha pontuação de crédito com Chase é xxxx, uma pontuação B, e contém informações erradas em relação ao histórico de minha conta.   O que Chase está fazendo comigo é irrelevante - nada mudará na minha vida. O que Chase está fazendo com as pessoas no meu quarteirão e no meu bairro é um crime. Chase está impedindo meus vizinhos, que não têm idéia de que suas pontuações de crédito estão sendo subestimadas, sofrem as consequências ao comprar uma casa, casa etc.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  \n",
            "\n",
            "Chase não discrimina ident\n",
            "======================================================\n",
            "Texto: A reclamação está entre xxxx xxxx xxxx e a placa de recompensas XXXX Chase XXXX que eu mantenho.   Comprei férias com este serviço em xx/xx/xxxx no valor de {$ 5500,00}, carregado integralmente no meu cartão de crédito XXXX Chase XXXX.   Isso comprado incluiu um resumo detalhado do Flight & Hotel. Transporte terrestre. E um plano de proteção de pacotes.   O custo do plano de proteção do pacote {$ 340,00} na política e política declarada na página 1, o direito de alterar ou cancelar por qualquer motivo antes da data programada (xx/xx/xxxx) para obter um reembolso total.   Cancelei a viagem em xx/xx/xxxxover o telefone com xxxx xxxx xxxx e foi verbalmente hoje os valores dos reembolsos que eu estava definido para receber. Em nenhuma ordem específica, o hotel deveria ser reembolsado no valor de {$ 2200,00} transporte terrestre deveria ser reembolsado no valor de {$ 300,00} e a passagem aérea deve ser reembolsada no valor de {$ 2700,00}. Eu poderia perder o prêmio do seguro conforme o esperado no valor de {$ 340,00} Os valores {$ 2200,00} e {$ 300.00} foram reembolsados ​​imediatamente; No entanto, o valor de {$ 2700.00} ainda não foi reembolsado, agora quatro meses completos depois.   As chamadas para xxxx xxxx xxxx xxxx foram feitas pelo menos 10 vezes. Solicitações on -line documentadas por meio de atendimento ao cliente estão disponíveis para revisão em xx/xx/xxxx. As chamadas telefônicas para agentes offshore e elevadas aos supervisores da equipe foram feitas, principalmente um correio de voz salvo no meu telefone celular datado de xx/xx/xxxx indicou que eu não deveria me preocupar (supervisor) que o reembolso e o valor confirmado de {$ 2700,00} estavam sendo processado.   Xx/xx/xxxx - escrevi uma carta para a empresa documentando minha reclamação.   Hoje, xx/xx/xxxx, registrei outro telefonema para um \"supervisor\" novamente sendo informado de que o valor do reembolso estava acima de sua autoridade, mas que ele estava elevando -o ao nível certo para me garantir que estava sendo processado.   Mais tarde, com quatro meses, a uma taxa de juros acima de 21 % - tenho certeza de que paguei bem mais de {$ 2700,00}, sou devido a esse dinheiro sem dúvida. A apólice de seguro que comprei foi comprada para me proteger dessa mesma questão. Talvez eu também tenha um problema com a XXXX Casualty Insurance Company, de acordo com o número da apólice XXXX\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  XXXX.   Por favor, verifique\n",
            "======================================================\n",
            "Texto: O dinheiro foi removido da minha conta sem minha autorização. Entrou na filial e assinou o cheque de seguro e foi informado ao endossar o cheque que os fundos seriam lançados em duas parcelas.   Ainda não há resolução após 7 meses. Estou constantemente recebendo as ruas, o que é muito frustrante e tenho sido muito paciente.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "\n",
            "Devido à sua situaç\n",
            "======================================================\n",
            "Texto: Eu era titular de conta no Chase Bank em xxxx. Wa. Fui a um caixa eletrônico e vi que tinha cerca de {$ 240,00} extra na minha conta. Eu pensei que era estranho, então imediatamente entrei no banco para consultar o caixa que também era gerente da filial. Eu disse a ela que achava que não havia dinheiro na minha conta, mas era uma possibilidade, pois eu tinha acabado de receber um grande depósito e não tinha certeza se eu tinha calculado mal minha conta. Expliquei isso a ela e ela me disse repetidamente que todos os meus pagamentos passaram e esse era o meu saldo disponível. Ela então começou a listar todas as minhas transações para garantir que não tenhamos falta de qualquer coisa. Depois que eu tinha certeza de que tínhamos passado por todos eles, eu disse que também poderia retirar o dinheiro. Eu me retirei em torno de {$ 240,00} em dinheiro e prossegui sobre o meu dia. Cerca de duas semanas depois, recebo uma notificação que minha conta estava em torno de {$ 240,00} exagerada e minha conta estava sendo fechada. Liguei imediatamente com pouca ou nenhuma ajuda. Fui a outra filial e o gerente da filial de outra filial disse que ele me ligaria para limpar o ar e resolver isso. Aqui estamos cerca de 6 meses depois, sem telefonema. Acredito que é responsabilidade dos bancos de Chase notificar e explicar toda a conta sobre informações desenhadas. Além disso, o fato de eu estar no banco quando retirei esse dinheiro e fui repetido pelo gerente da filial que eu estava limpo e estava mais do que bom para retirar esse dinheiro.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Estou perguntando sobre as evidências que a Chase deve sentir que é por lei para congelar minha conta e minha aposentadoria e fundos de seguridade social e desemprego. Eu preciso saber quais medidas eles estão tomando e estão tomando alguma para corrigir essa situação imediatamente? E qual é o prazo que eles buscam continuar mantendo meus fundos. Eu sei que isso é uma vingança porque xxxx xxxx e seu escritório executivo feito como uma tática de discriminação xxxx e xxxx corporativo contra mim am xxxx xxxx consumidor.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "A Chase não pode congel\n",
            "======================================================\n",
            "Texto: Eu tenho uma Mortga GE com o JPMorgan Chase & Co e solicitei assistência hipotecária que foi negado em xxxx xxxx, xxxx. O agente que me informou sobre essa decisão me disse para fazer algum pagamento antes do xxxx xxxx, xxxx. No xxxx xxxx, xxxx, eu estava com 4 meses de vencimento na hipoteca e o agente me aconselhou a fazer o xxxx xxxx, xxxx para manter o passado devido a menos de 5 meses. Fiz um pagamento em parte {$ 2000.00} na parcela da hipoteca devido a aproximadamente {$ 3700,00} por princípio, juros e garantia. Fiz um pagamento adicional de um mês vencido no xxxx xxxx, xxxx. Chase devolveu os dois pagamentos para mim e não aceitou nenhum pagamento inferior a 3 meses vencido ou {$ 11000,00}. Eu havia pago a eles aproximadamente {$ 5700,00} para recuperar o atraso nos valores vencidos e refudi algo menos que {$ 11000,00}. Eu nunca experimentei nenhum credor recusando o pagamento, a menos que fosse um valor mínimo. Essa prática é uma desculpa para permitir que eles prosseguem para a execução duma hipoteca e obtenham um lucro inesperado nessa hipoteca que foi comprada por perseguir a FICA como resultado da falha do Washington Mutual (WAMU) em xxxx por menos de centavos por dólar. No XXXX, Chase se recusou a modificar minha hipoteca sob Hamp pelo mesmo motivo e eles falsificaram o endosso de Wamu e Robo assinaram a transferência pela qual foram penalizados por várias agências governamentais. O Chase Fiounly modificou minha hipoteca em xxxx depois que eles resolveram uma ação judicial que eu arquivei contra eles e me paguei danos de {$ 15000,00}. Chase finge fornecer assistência hipotecária e me designou um representante pessoal que eu nunca posso entrar em contato porque ela não está disponível. Quando entro em contato com meu representante pessoal, ela nunca está lá e outra pessoa precisa atender minha ligação. Quase todos os supostos representantes de atendimento ao cliente têm um sotaque estrangeiro e não têm autoridade para tomar nenhuma decisão. Isso é um ardil de Chase que eles praticaram no XXXX quando apresentaram uma execução hipotecária contra mim depois que o agente nos disse para não fazer pagamentos na hipoteca até que a modificação tenha sido concluída. Naquela época, eu nem estava vencido por nenhum valor. Sou XXXX anos e preciso de assistência das práticas de coleta pré -datativa Chase.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Não posso gerenciar as finan\n",
            "======================================================\n",
            "Texto: Em xx/xx/2017, depositei {$ 960,00} no novo atm perseguido em xxxx xxxx, Florida. A máquina fez um som de zumbido por alguns minutos e depois afirmou que havia um erro e o caixa eletrônico deu um recibo.   Sequência # xxxx \"Depósito ao CHK Acct xxxx Este dispositivo teve um problema técnico. Para confirmar que sua última solicitação foi concluída corretamente, visite um banco de perseguição ou ligue para xxxx ''. Não saí do caixa eletrônico por cerca de uma hora, enquanto Liguei para o número e falei com um representante que alegou que ela creditaria minha conta {$ 960,00} dentro de uma hora enquanto a questão estava sob investigação do departamento de reivindicações. O dinheiro não foi depositado no dia seguinte, então tive que ligar de volta e as reivindicações O departamento não viu nenhum registro disso me informando. Eles creditaram minha conta os {$ 960,00} e receberam informações de mim, caso tenham que entrar em contato comigo por qualquer dúvida sobre a investigação. Xx/xx/xxxx, {$ 960,00} foi retirado da minha conta sem nenhum aviso do departamento de reivindicações do Chase Bank. Liguei e me disseram que não havia dinheiro no caixa eletrônico e a reivindicação foi negada. Também me disseram que eles não olhavam para a câmera. Entrei na filial e o gerente da filial, xxxx, disse que ela pessoalmente o procuraria. Ela ligou para o departamento de reivindicações e enviou por fax sobre o recibo que afirmava que houve um erro de caixa eletrônico no meu depósito. O \"Departamento de Escalações\" afirmou que resolveria o problema em 24 horas. Eles não. O gerente da filial ligou de volta e disseram que nunca disseram isso. Até o gerente da filial afirmou que nunca viu tanta incompetência em 15 anos em seu trabalho. O departamento de escalações alegou que levaria 7 a 10 dias úteis. Não ouvi nada e liguei hoje, falei com 4 pessoas em 20 minutos e finalmente conversei com \"xxxx\", que afirmou que era contra a política da empresa para dar um agente #. XXXX me disse que a reivindicação foi negada, que eles olharam para o vídeo e não viam nada que seja uma mentira. Então, o cara que verificou o caixa eletrônico roubou {$ 960,00} de uma mãe solteira que nunca havia me elaborado ou perseguindo escalados departamento de escalados me enganou de {$ 960,00}. Não entendo como isso é remotamente aceitável. XXXX me disse que seu supervisor me ligaria de 24 a 48 horas para me dizer o que ele já disse. Eu disse a ele que queria isso resolvido e ele disse que o problema foi resolvido, mas eu não queria aceitá -lo.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  Aqui está a minha queixa\n",
            "======================================================\n",
            "Texto: Minha conta foi cobrada por taxas de fundos insuficientes XXXX nesta manhã. No fim de semana, minha taxa XXXX foi retirada e me deixou {$ 98,00} Overrawn. (Eu não sabia que o banco aconteceu no fim de semana) Na terça -feira de manhã, o Banco Chase optou por me cobrar pelo excesso de cargas menores xxxx, xxxx delas {$ 22,00}, que é menor que a taxa de overdraw, xxxx por menos de {$ 35,00 }, que é menor que um dólar sobre a taxa de overdraw em si e xxxx por {$ 55,00}. O raciocínio que eles me deram é que eles lidam com as maiores acusações primeiro e algo sobre cobranças pendentes versus acusações limpas. Nenhuma das acusações está pendente para mim, uma vez que removem o dinheiro da minha conta no segundo lugar. Perseguir. Além disso, as taxas menores teriam sido cobertas até a taxa XXXX - portanto, a taxa de fundos insuficiente era válida para xxxx, mas não para as taxas menores. A taxa XXXX aconteceu em um fim de semana, quando não deveria haver bancos de qualquer maneira. Liguei para Chase para reverter as taxas e elas recusaram. Falei com xxxx pessoas diferentes, sendo o último supervisor xxxx que 1) disse que ela tentou reverter, mas o sistema não a deixou porque eu passei pelo meu limite de cortesia este ano 2) perguntei a ela qual agência regulatória eu poderia Reclamar porque isso é roubo corporativo, e ela me disse que é a XXXX com a qual eu reclamaria. Pedi a ela novamente que me dissesse quem regula os bancos e ela disse que entendia e sentia muito, mas não recomendou uma agência regulatória.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Eu tenho minhas contas com Chase há anos. Recebi uma carta afirmando que eles decidiram fechar minha conta e liguei o que naquela época foi informado de que eles não precisavam divulgar o porquê e eu pedi para falar com um gerente e disseram que ninguém precisava divulgar por que eles decidiram encerrar nosso relacionamento. Fui então colocado em espera por uma hora, que naquele momento eu desliguei o telefone. Eu até tenho meu empréstimo de automóvel com eles. Sou um ótimo cliente e sinto que não há razão para que minha conta seja fechada sem explicação.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Chase Mortagage não atualizou as informações da minha conta corretamente. O relatório de crédito mostra que estive atrasado nos meus pagamentos de hipoteca para xxxx e xxxx deste ano, o que não está correto. Eu estou atualizado em meus pagamentos de mortagagem desde xxxx deste ano.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Exigirei que Chase Mortagage cor\n",
            "======================================================\n",
            "Texto: Chase não respondeu à atividade suspeita que encontrei nos cartões abertos e fechados em meu nome, de acordo com meu relatório de crédito xx/xx/xxxx. A placa XXXX aberta no xxxx usou um endereço de email de xxxx e um endereço xxxx xxxx xxxx xxxx ct que não era meu ou em que eu vivi. O XXXX XXXX possui esse e -mail e usou este cartão para desviar os fundos de Chase para si mesmo, possivelmente posando como o fornecedor do comerciante. Por favor, revise as transações XXXX neste cartão a partir da data em que foi aberta no XXXX até que eu o encontrei em xxxx e não sabia nada sobre isso. Você está sendo enganado e enviando dinheiro para uma pessoa e não para xxxx. Revise as transações de xxxx para xxxx.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "A Chase não responderá automatic\n",
            "======================================================\n",
            "Texto: Trabalhamos com o banqueiro hipotecário xxxx xxxx para obter um empréstimo à habitação do JP Morgan Chase Bank. Xxxx xxxx, 2015, xxxx xxxx enviou um email para a Chase \"Taxa aprovou a gerência em 3,5 % com pontos de xxxx '' para um empréstimo à habitação de 30 anos (apoiando xxxx # xxxx). Xxxx xxxx, 2015, assinou o empréstimo do\" bloqueio \"-in-in) Contrato '' e os devolveu a xxxx xxxx para bloquear a taxa de empréstimo em 3,5 %, taxa de originação de 0 %, pontos de desconto de 0 %e taxa estendida de bloqueio de {$ 0,00}. XXXX Suporte Doc # xxxx e # xxxx) xxxx xxxx, 2015, recebemos um e -mail do xxxx xxxx nos parabenizando pela aprovação do Chase Grant, o que significa que \"Chase pagará xxxx por [nosso] custo de fechamento.\" # Xxxx) recebemos uma estimativa de boa fé (GFE) somente quando solicitamos xxxx xxxx, 2015. Ao revisar, notamos pontos de desconto. Por conversa por telefone, xxxx xxxx aconselhado que fomos cobrados pontos de desconto no empréstimo devido à aprovação Para Chase Grant. Essa condição não foi divulgada a nós com nenhuma comunicação sobre o Chase Grant nem mencionou a nós até que vimos os pontos no GFE e a trouxemos para questionar. Como xxxx xxxx disse que apenas a gerência pode remover os pontos, nós Entramos em contato com o gerenciador xxxx xxxx, xxxx xxxx. Fizemos vários telefonemas; todos foram devolvidos. Entramos em contato com um xxxx xxxx, que prometeu uma chamada de volta de alguém para resolver o problema, mas novamente não fizemos receber uma ligação de volta. Finalmente conseguimos GE T A Hold of XXXX XXXX em xxxx xxxx, 2015 e por sua solicitação, enviou uma prova por e -mail da taxa aprovada de 3,5 %, XXXX Points de desconto. Em xxxx xxxx, 2015, recebemos um e -mail de xxxx xxxx dizendo que \"conversou com xxxx, ele honrará o ponto xxxx. Você ainda recebe o chase xxxx. Os documentos foram enviados para xxxx xxxx para assinatura. Enquanto estava no telefone com xxxx xxxx com o cofrow para providenciar o notário móvel para assinatura do mesmo dia, xxxx xxxx nos informou xxxx xxxx apenas deu o word mutuário Perguntamos a xxxx xxxx o que aconteceu, mas ela sugeriu que conversássemos com xxxx xxxx. Xxxx xxxx explicado por telefone que, na conversa com o processador de empréstimo, o processador determinou que o mutuário não podia assinar e imediatamente colocou uma extensão de bloqueio na taxa, Tornando o empréstimo original vazio. Isso era simplesmente falso, pois já estávamos agendando a assinatura com custódia na próxima hora. XXXX XXXX não comunicou essa decisão conosco antes de colocar a extensão de bloqueio. Devido ao xxxx xxxx 's' s negligência e o Ações tomadas sem consentimento formal do mutuário, o empréstimo não financiou e o custódia não foi capaz de fechar. Além disso, fomos cobrados {$ 830,00} pelo empréstimo como uma taxa de extensão de bloqueio, inconsistente com o \"Contrato de bloqueio\", que declara uma taxa estendida de bloqueio de {$ 0,00}. XXXX XXXX, 2015, contatamos Xxxx xxxx para resolver as preocupações com o crédito do Chase Grant aplicado aos custos de fechamento no \"HUD-1 final\" (apoiando o Doc # XXXX). De acordo com a resposta por e -mail recebida de xxxx xxxx, ele aconselhou que a concessão foi \"calculada nos números no HUD na linha xxxx\" '(suportando xxxx # xxxx) na revisão, a linha xxxx é o {$ 830,00} (0,25 % de pontos carregados incorretamente pela taxa de bloqueio estendida). Não houve crédito de {$ 1400,00} para a concessão do Chase. Entramos em contato com xxxx xxxx, que aconselhou que a concessão foi aplicada aos pontos de desconto e depois prosseguiu para trivializar nossas preocupações e desviar a conversa dizendo que \"grande negócio '' temos. Ele nunca reconheceu seus erros nem abordou a questão de desinformar e enganar -nos em termos de empréstimo. Devido à negligência e à maneira não profissional de xxxx xxxx e seus colegas, pagamos {$ 2300,00} por taxas excedentes de termos.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta:  pedido que esses gastos extras sejam\n",
            "======================================================\n",
            "Texto: Em xx/xx/2019, minha conta não passou quando estava no automóvel. Entrei em contato com a empresa e eles disseram que viram o pagamento tentar sair do banco, mas foi rejeitado. Eles nunca me notificaram da rejeição e me deram esse atraso no pagamento. Agora não posso obter uma hipoteca por causa dessa marca em meu crédito e a empresa não me ajudará com essa situação injusta.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Inundação xx/xx/xxxx Ocorreu. Entrei em contato com xxxx xxxx (provedor de seguros). Eles me disseram que eu poderia tirar qualquer coisa molhada. Peguei roupas, toalhas, itens arruinados, móveis, etc. Eles enviaram um ajustador. Em sua visita, ele disse que estava me aprovando por uma verificação de construção de {$ 8000,00} para começar. E para seguir em frente e entrar em contato com minha empresa de hipotecas, porque eles precisariam endossar também. Ele também me disse que eu poderia tirar qualquer coisa molhada. Ele me informou que eu não precisava ter um empreiteiro de licença para trabalhar na casa. Eu poderia escolher quem eu queria ou fazer isso sozinho. Liguei imediatamente para Chase, disse a eles que inundei, tenho um cheque chegando e perguntei o que eu precisava fazer. A mulher me disse para escrever um número de rastreamento, assinar o cheque e enviar a eles para que eles pudessem endossá -lo. Nesse momento, eu não sabia que eles mantinham o dinheiro e me trouxeram. Não houve nada dito sobre um processo ou qualquer outra coisa. Xx/xx/xxxx {$ 8000.00} Verificação cortada por xxxx. Xx/xx/xxxx entrou em concordância com xxxx xxxx para rasgar e instalar. XX/XX/XXXX Demo/remoção iniciada por xxxx xxxx. {$ 5000.00} xx/xx/xxxx Eu enviei check -in para perseguir. Chase xx/xx/xxxx Corte meu {$ 8000,00} Verificação de inicialização para reparo de construção. REPAROS ELÉTRICOS XX/XX/XXXX Madeers não poderiam funcionar {$ 220,00} xx/xx/xxxx a secagem iniciada. {$ 3000.00} xx/xx/xxxx Reparos elétricos feitos {$ 170,00} xx/xx/xxxx nova bomba sépticos instalada {$ 700.00} xx/xx/xxxx i Enviei no {$ 72000,00} verificação de xxx para perseguir, endors. Conversei com xxxx xxxx lá e ela disse que envie o cheque endossado e que o dinheiro foi colocado em uma conta de garantia restrita. XX/XX/XXXX A declaração de intenção de reparar via site xxxx, onde eu conectei primeiro e comecei a ver os formulários que Chase tinha online. Xx/xx/xxxx entrei em contato com xxxx e disse que os formulários tinham que ser concluídos. Ele disse que não estava ligado. Eu disse a ele que entraria em contato com Chase. Contatou Chase. Xx/xx/xxxx disse ao XXXX que conversei com o Chase e eles disseram que, desde que ele fosse licenciado, você poderia deixar as informações de vínculo. Ele disse que o drywall não é o sub -trabalho licenciado. Eu assegurei que ele o receberia seu dinheiro porque Chase estava ciente da minha situação. Xx/xx/xxxx Intenção enviada para reparar o formulário xx/xx/xxxx entrei em contato com xxxx e pedi que ele completasse XXXX e a renúncia de penhor contratados e disse que se ele pudesse preencher isso, Chase sabe que não preciso de um empreiteiro geral (para fazer drywall trabalhar ) Xx/xx/xxxx Eu pedi xxxx para a papelada novamente e eu disse que tudo bem sobre a licença e a ligação. Eu conversei com o Chase sobre isso. XX/XX/XXXX-XX/XX/XXXX Instalação Drywall concluída {$ 6500.00} XX/XX/XXXX XXXX Recibos de dispositivo carregados. {$ 4700.00} xx/xx/xxxx Primeira inspeção solicitada por mim. Cancelado por Chase quando? Letra xx/xx/xxxx enviada da Chase dizendo que recebeu minha solicitação de exceção, mas não pode aprová -la até que recebam um relatório de inspeção mostrando o status dos reparos. Tile de XX/XX/XXXX Concluído em Bath {$ 630.00} xx/xx/xxxx enviou uma carta ao Chase XXXX por suas instruções, solicitando renunciar aos contratados renúncia e cópia dos contratados Licença/Seguros e Bonding. Eu conversei com o Chase naquele dia e me disseram que tudo ficaria bem, apenas para enviar a carta de solicitação. Xx/xx/xxxx madeira entregue {$ 1700,00} xx/xx/xxxx-xx/xx/xxxx pintando {$ 1800.00} (pago por suprimentos de tinta antes de xx/xx/xxxx {$ 1200.00}) xx/xx/xxxx 2nd inspeção Quando descobri que o primeiro foi cancelado, eles disseram porque o inspetor tentou entrar em contato comigo três vezes. Xx/xx/xxxx-xx/xx/xxxx pisos de concreto pintados {$ 440,00} xx/xx/xxxx e suprimentos de porta adquiridos {$ 1100.00} xx/xx/xxxx-xx/xx/xxxx e osors instalados {US $ 790. 790.00 XX/XXXX GRAFT E SUPLIMENTOS DE PORTA {$ 120.00} xx/xx/xxxx 2ª inspeção cancelada por Chase. Vi isso online. Não tinha notificação. XX/XX/XXXX As formas não foram dispensadas por Chase. Conversei com xxxx xxxx perguntou por que a inspeção foi cancelada. Ela disse (a segunda vez que me disseram isso) que as notas dizem que a inspeção foi cancelada porque eles tentaram entrar em contato comigo. Isso é o que me disseram nas duas vezes. Ela disse que o motivo era provavelmente porque eu ainda não tinha um empreiteiro licenciado, mesmo que fosse observado o contrário. Eu nunca recebi ligações de inspetores ou mensagens, apesar de terem dito que eu tinha pelo menos 6 ligações referentes. Pedi para falar com um supervisor, obtenha xxxx xxxx, especialista em resolução/supervisor de conta. Precisava de uma licença e a solicitação nunca foi aprovada. Ela disse que você pode solicitar o que quiser, não significa que você vai conseguir. Esse é o ponto de um pedido. Tudo dito em questão e tom condescendente. Eu informei que Chase havia me dito para fazer o pedido em primeiro lugar e me garanti que estava bem para a minha situação. XX/XX/XXXX 3rd Inspeção solicitada xx/xx/xxxx letra enviada por Chase dizendo que recebeu a solicitação de exceção, mas não pode aprovar até que tenham pago em faturas completas e recibos válidos excedendo {$ 28000,00} lançados até agora. O inspetor xx/xx/xxxx chamado para marcar uma consulta. Xx/xx/xxxx banheiro e pia no banho completo instalado xx/xx/xxxx letra enviada por correio que disse que aprovou minha solicitação de exceção e me enviará uma verificação de fundos de reivindicação de seguro por xx/xx/xxxx. O próximo cheque que recebi foi xx/xx/xxxx para {$ 790,00}. XX/XX/XXXX teve inspeção, foi considerado 70 % completo por terceiros contratados pela Chase. Xx/xx/xxxx Eles enviaram uma carta com meu cheque para {$ 790,00} que disseram que enviarão o restante dos meus fundos. Eles recebemos um relatório de inspeção final, mostrando que os reparos na propriedade estão em pelo menos 90 % completos. Xx/xx/xxxx conversou com xxxx xxxx da Chase e ela disse que os reparos não estavam completos. Eu estava em 70 %. Eles ainda estavam segurando {$ 50000,00}. Perguntei o que precisava ser feito para 100 % de inspeção e ela disse pisos, armários, acessórios, pontos de venda. Sem dinheiro até que eles obtenham uma licença de contratado e renúncia a penhor. XX/XX/XXXX Os contratados com contratados renúncia e contrato xxxx e cartas solicitando reembolso e recibo final. Xx/xx/xxxx Eu havia solicitado um cheque para ele da Chase. XX/XX/XXXX Converso com xxxx em xxxx. Disse que eu precisava de uma licença de contratados ou as coisas poderiam ser mais simples se eu mudar para o autocontratador. Eu disse que nunca tinha ouvido saber que era possível, e a primeira vez que sabia que era uma opção. Ela disse que era possível se eu tivesse uma hipoteca atual com eles. Embora se ela me mudasse para a auto-contratação, eu não poderia usar o site de cheques de seguro e tiver que pagar tudo fora do bolso primeiro e depois ser enviado quando terminar totalmente. Eu disse que não me sinto confortável em mudar, preciso falar com meu advogado com o que devo fazer porque a) preciso de um registro eletrônico de tudo e a autocontrase depende de fax e correio b) Como posso ser assegurado o dinheiro virá até mim no final? Ela disse que iria ligar para XXXX para ver se, em vez de uma licença, ele poderia produzir prova de seguro / fiança. Eu disse, sinta -se livre. Ela disse que funcionaria para liberar meus fundos. Xx/xx/xxxx conversou com xxxx xxxx no chase porque eles lançaram {$ 450,00} Eu perguntei o que foi, foi para recebimentos xxxx de suprimentos de mitigação. Eu disse a ela que pensei que me disseram que não estava recebendo mais dinheiro até terminar, porque ainda tinha muitos recibos não pagos. Ela disse que se eu pudesse enviar o cheque cancelado XXXX XXXX {$ 6500.00}, que eles me reembolsariam por isso, mas não lançariam mais fundos. Ela também disse que as notas diziam que XXXX XXXX havia sido chamado, mas nenhuma resposta em xx/xx/xxxx. XX/XX/XXXX Chase me ligou para me informar que eles cancelaram minha solicitação de cheque para pagar xxxx xxxx {$ 6500.00} desde que eu enviei um recibo que foi pago. Piso xx/xx/xxxx colocado xx/xx/xxxx armários instalados. Pisos xx/xx/xxxx a serem selados.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Enviei todas minhas not\n",
            "======================================================\n",
            "Texto: Chaserelief of Stay Aplicação é completa negligencie as informações de propriedade fornecidas ao tribunal de falências não é meu homechase também observa que nenhum aviso foi enviado sobre a falência. Para mim, em nenhum momento no último trimestre de xxxx, conforme necessário. No entanto, Chase sabia aparecer e solicitar alívio da permanência e concedido com base em uma aplicação falsa negligente com a propriedade errada - não na minha casa. A avaliação concluída por/contratada e aprovada por Chase durante a falência foi XXXXXXXXCHASE vendida para casa por um pouco mais de xxxx milit\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: ares. Normalmente, Chasehonra solo\n",
            "======================================================\n",
            "Texto: Um cartão de crédito Chase/XXXX foi aberto sem o nosso consentimento. Todos os materiais, incluindo um cartão, foram enviados para nossa casa. Chamamos Chase para alertá -los e, em vez de nos preocuparmos com essa misteriosa abertura, eles perguntaram várias vezes se tínhamos certeza de que não queremos essa conta. Eles disseram que o XXXX o abriu. Chase disse que fechou essa nova conta de cartão e disse que eles nos enviam material sobre roubo de identidade pelo correio. Muito suspeito.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  Providenciaremos as providên\n",
            "======================================================\n",
            "Texto: Em xx/xx/2020, fui verificar minhas transações nos últimos 3 meses e vi o XXXX foi debitado na minha conta em xx/xx/2020. O nome da loja foi listado como xxxx xxxx xxxx. Notifiquei imediatamente o Chase Bank e os informei da transação que não foi feita por mim, ou autorizada a fazê -lo por mais ninguém. Disseram -me que meus fundos estariam na conta dentro de 12 a 24 horas que um carro novo receberia que o cartão antigo precisava ser cancelado. Expliquei que estava fora da cidade e precisava não cancelar meu cartão até voltar para casa e cancelaria o cartão na segunda -feira seguinte ao retornar. Cancelado O cartão na segunda -feira fez um acompanhamento no xxxx de xxxx, nenhum fundos foi colocado em minha conta por Chase, liguei de volta mais duas vezes para perseguir informações e, novamente, foram informados que os fundos estariam na conta dentro de 24 horas. Chamado de volta xx/xx/xxxx para perseguir um nome representativo xxxx declarou que minha reclamação foi negado, me informou para notificar a polícia e não há nada que o Chase Bank possa fazer e nenhuma informação estava disponível para apoiar o motivo da negação. Chamado Chase XX/XX/XXXX falou com XXXX A Rep for Chase e foi informado de que uma carta foi enviada do xxxx de xxxx e novamente me disseram que eu teria que receber meu dinheiro da pessoa que o roubou. Disseram -me que eu tenho alerta de fraude e proteção de fraude que me notifica quando uma transação é feita, mas não recebeu uma notificação por essa grande quantidade. Novamente, os representantes não tiveram explicação sobre o porquê.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Em xxxx xxxx, 2016, notei que xxx dos XXXX Accts em que sou um assinante tinha xxxx depósitos pendentes. O final do Acct # xxxx foi minha filha Acct. O primeiro depósito foi de sua escola por {$ 290,00} que estávamos esperando. O outro xxxx foi verifica xxxx por {$ 980,00} e o outro para {$ 980,00} de xxxx um programa de verão para o qual trabalhou durante o verão. Os depósitos pendentes não me pareciam bem, então liguei para XXXX XXXX XXXX, que estava no comando do programa durante o verão. Perguntei a ela se eles deviam a minha filha algum dinheiro no verão e ela afirmou que é uma possibilidade e ela verificará quando entrou no escritório. Enquanto isso, liguei para Chase e os informei de que achava que os cheques XXXX eram fradulentos e para não deixá -los não passar pela conta e perseguir a resposta representativa era que minha filha tinha que ligar. Eu disse, a eles que sou uma pessoa autorizada na conta e eles devem poder conversar comigo. Xxxx xxxx ligou de volta e me perguntou se eu poderia enviar uma cópia das verificações xxxx para que ela possa enviar para as contas a pagar. Mais tarde, à noite, um xxxx xxxx, o diretor de recursos humanos me ligou de XXXX e me perguntou se eu achava que minha filha havia cometido fraude porque ele havia ligado Chase e eles afirmaram que minha identificação de usuário e senha da filha era usada para depositar os cheques. Também expliquei a ele que um formulário de depósito direto havia sido entregue a xxxx para que ela tivesse depósito direto e xxxx nunca fez o depósito direto, mas deu seus cheques em papel e onde estava a forma e em quem se esteve. Liguei para o Chase de volta para ver qual dispositivo, computador ou célula foi usado para depositar os cheques. Chase me transferiu para xxxx pessoas diferentes para me dizer apenas que a conta estava sendo investigada. Eu também disse ao THM que minha filha havia perdido seu cartão de débito e qualquer um poderia ter seu cartão de débito e formulário de depósito direto que xxxx nunca usou para direcionar DePosti Sua folha de pagamento. A última pessoa com quem conversei me disse que todos os meus Accts estariam fechados e Chase estava terminando todo o relacionamento comigo e com minha família. Afirmei por que eles fechariam todas as minhas contas quando apenas uma conta tivesse atividade fradulenta. Eles afirmaram que não poderiam me dar mais informações e o relacionamento com Chase estava terminando. Perguntei quando eu poderia ter meu dinheiro dos outros Accts e eles afirmaram que não poderiam me dar nenhuma informação novamente. Eu não entendo por que eles estão segurando meus outros acts. Mais tarde, entrei em um local Chaswe no XXXX e XXXX para conversar com alguém e eles me disseram que poderiam me fornecer qualquer informação sobre os Accts. Esperei verificar meu banco on -line diariamente para ver se ocorreram alguma alteração e, em xxxx, notei que minha filha não parecia mais. Liguei para o Chase e eles declararam que fecharam o ACCT e um cheque de caixa está sendo enviado para o depósito correto por {$ 290,00} a partir de hoje, não recebemos o cheque de caixa. No XXXX, recebi uma carta da Chase afirmando que não tínhamos cumprido as informações que Chase havia solicitado. Liguei para Chase e eles declararam que era uma carta padrão que não faz sentido. Perguntei quando meus outros Accts seriam lançados porque eles não tinham nada a ver com o ACCT fradulento. Expliquei que um dos Accts era minha mãe que era idosa e eles haviam parado todas as funções de nossas vidas porque ela precisa de seu remédio e nenhum de nós tem acesso ao dinheiro. A partir de hoje, xxxx/xxxx/2016, eles ainda estão segurando os acts não fradulentos e eu não recebi cheque de caixa que eles declararam que enviaram o ACCT fradulento. Eu realmente preciso resolver esse problema porque minha mãe alugou, outras contas e A medicina precisa estar cuidando e estou na névoa de me mover. Eu sinto que Chase agora está retaliando contra mim e minha família sem nos dar Prooof de que fizemos algo errado.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Eu havia depositado cinco peças de pagamento em uma filial do Chase Bank, localizada em xxxx xxxx xxxx, xxxx ca xxxx em xx/xx/xxxx. Cada ordem de pagamento tem uma quantia de {$ 1000,00}, um total de {$ 5000,00}. Tenho muitas evidências para processar o banco por atividades suspeitas. Primeiro, o caixa eletrônico recebeu minhas ordens de pagamento, mas não pôde processar e congelar na tela de carregamento. Entrei no banco e pedi ajuda de um dos associados do banco. Ela se recusou a sair e me ajudar com o meu problema, apesar dos clientes que estavam esperando sua vez. Ela me disse para pressionar o botão de cancelamento. No entanto, as ordens de pagamento não foram dispensadas, apenas o recibo dizendo que a transação não poderia processar. Entrei para obter ajuda novamente, o Associado do Banco me ajudou a fazer uma ligação para abrir uma investigação. Ela me disse que a investigação leva 60 dias. Eu esperei pacientemente em meio à pandemia. Eles não sabem como minha família luta para pagar pelas contas. Nós realmente precisamos desse dinheiro para pagar o aluguel. Enquanto esperava, recebi uma carta do banco, dizendo que eles viram a transação feita por mim em xx/xx/xxxx e que era legítima e autorizada. Eu esperei até cerca de 40 dias, liguei novamente para verificar meu caso, apenas para ouvir uma declaração chocante. Eles fecharam meu caso !!! ?? Sem nenhum aviso e conhecimento dos clientes, é inaceitável de como eles negam sua responsabilidade. Eles disseram que investigaram e não conseguiram encontrar todas as cinco peças das minhas ordens de pagamento. Perguntei sobre a carta de confirmação que eles me enviaram sobre como receberam minhas ordens de pagamento, para ouvir uma verdade de perturbação que eles enviaram uma carta errada !! ?? Estou muito estressado por suas mentiras flagrantes. Eu fiz e gravou ligações telefônicas, conversando com o supervisor bancário. Literalmente, toda vez que eles desligavam o telefone enquanto ainda estávamos conversando! Eu sei que está no meio da pandemia, mas foi uma coisa muito imprudente a fazer com seus clientes. Eu pedi para ver a câmera de segurança bancário/atm, eles me disseram que não têm a autorização e que eu tenho que ir ao departamento de polícia para registrar um relatório da polícia. Eu fui lá, mas o departamento de polícia está fechado devido à pandemia. Liguei então para o número no aviso da porta da frente. Uma mulher me cumprimentou e eu contei minha história. Ela então me disse que a polícia não resolve esses casos, a menos que haja um assalto ou assassinato, e que meu caso é apenas um caso civil. O banco tem todo o direito de abrir a câmera de segurança para ver a filmagem. Ela me disse para falar com o supervisor do banco e pedir uma intimação. Minha vida foi provocada por causa de sua irresponsabilidade. Não desejo fazer disso um grande negócio, o que pude. Estou registrando essa reclamação ao CFPB como minha última ação antes de fazer disso um grande negócio, se o banco continuar escondendo e negando sua responsabilidade. Estou implorando para você, o CFPB para me trazer justiça e mostrar a eles suas irregularidades. Muito obrigado.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Fui enganado por uma conversa telefônica do Chase Bank sobre o adiantamento que paguei para um refinanciamento da minha hipoteca atual. Disseram -me que o depósito era reembolsável. Uma inspeção nunca foi concluída na propriedade, mas o avaliador disse que eu tinha que substituir uma porta interna A, piso de azulejos e consertar o trabalho de spackle no teto do banheiro. Nunca houve uma compra ou contrato de refinanciamento dado a mim declarando os termos do meu adiantamento ou permitir quaisquer contingências. Disseram -me que meu pedido foi negado, a menos que eu faça reparos que o avaliador não pôde confirmar se os reparos eram necessários. O idioma que eles usavam era que eles \"podem ser\" necessários para reparar, mas não exigiam.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "\n",
            "Nesse caso, aconsel\n",
            "======================================================\n",
            "Texto: Notei hoje no meu relatório de crédito que existem xxxx novas consultas de chase. Até agora, não vejo nenhuma nova conta e quero garantir que o Chase não seja aberto ou feche as contas fraudulentas.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Bom, a primeira coisa\n",
            "======================================================\n",
            "Texto: Eu tenho um cartão de crédito Chase Marriott Rewards. Em xx/xx/xxxx xxxx carregado {$ 320,00}. Entrei em contato com XXXX porque eles não deveriam receber o pagamento automaticamente pela minha conta XXXX e porque o total de Chraged era para minha conta e também para a conta da Mothers XXXX. Em xx/xx/xxxx, minha mãe e eu ligamos para xxxx, onde eles enviaram uma confirmação de que iriam reembolsar meu {$ 320,00} e, em seguida, passamos a pagar pelas duas contas separadas a uma taxa diferente em um cartão de crédito diferente. O número de confirmação do XXXX é xxxx e a representação que nos ajudou foi xxxx. Como o reembolso nunca passou, estendi a mão para perseguir essa cobrança. Chase pediu que eu enviasse à prova de documentação. Então, em xx/xx/xxxx, eu requei com xxxx no perseguir a documentação de disputas. Em xx/xx/xxxx, depois de esperar mais de uma semana para que a documentação seja enviada por e -mail, enviei por e -mail e fax uma cópia da minha disputa, além de uma cópia do extrato bancário do meu Padres e do Nubmer e nome da confirmação xxxx e nome para o meu reembolso. Xx/xx/xxxx que aprendi depois de jogar a etiqueta de telefone por uma semana que eles negaram minha reivindicação, porque eu escureci o número da conta do meu pai por sua proteção. Depois de uma hora de telefonema em torno de xxxx com xxxx xxxx um supervisor de disputa no Chase, ele se eu for re-submitido, minha prova com pelo menos os últimos 4 dígitos da conta de Myy Faters, mostrando que dentro de 24 horas eu teria uma resposta. Em xx/xx/xxxx em xxxx, reenviei minha prova. Aparentemente, isso era uma mentira, porque em xx/xx/xxxx em xxxx, chamei Chase novamente para obter algum tipo de atualização. Eu falei pela primeira vez com xxxx, que claramente parou de me ouvir no meio da minha conversa e depois me disse que não poderia estar conectado de volta a xxxx e, em vez disso, me enviou a um supervisor chamado xxxx em xxxx que me disse que o quadro de 24 horas não foi possível. Que ela examinaria minha chamada xx/xx/xxxx e voltaria para mim. Perguntei a ela se havia alguma atualização ou qual era o meu status e ela não tinha resposta apenas me disse para esperar 30 dias. Eu nunca ouvi de xxxx novamente. Então, esperei os 30 dias e, em xx/xx/xxxx, em xxxx, chamei Chase para uma atualização. Fui transferido do XXXX para, eventualmente, um novo supervisor em disputas chamado xxxx que me contou em xx/xx/xxxx Minha disputa foi negado porque não enviei prova. Ele viu minha tentativa no XX/XX/XXXX, mas houve um erro aparente em transmitir meus anexos, mas ninguém de Chase pensou em ligar ou me enviar um e -mail me contando sobre esse problema. Eles também nunca pensaram em ligar ou enviar -me um e -mail depois que minha reivindicação foi novamente negada. Então XXXX me disse para tentar re-submissão novamente minha prova, mas desta vez fax. Então, em xx/xx/xxxx, enviarei um fax, outro email e uma carta certificada com toda a minha prova. Também estive em contato com um advogado para ver quais opções tenho. Continuo enviando provas e Chase continua negando minha reivindicação dizendo que eles não têm provas. Não sei mais o que fazer para provar que essa é uma disputa legitmate, e o {$ 320,00} é muito dinheiro para mim e não acredito que seja responsável por pagar esse total. Agora sinto que Chase está roubando de mim e não quero que eles atacem minha excelente pontuação de crédito por causa de sua negligência.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Eu tenho um cartão de crédito com Chase. Eu também tenho xxxx & am na escola em período integral, então minha esposa concordou em pagar nossas contas. Esta é a segunda vez que ela está atrasada fazendo esse pagamento em 4 meses, o que é raro. Enviamos um pagamento por correio no xxxx do XXX XXX, o pagamento foi devido no xxxx, mas a empresa não o recebeu até o XXXX. Eles cobraram uma taxa de atraso. Liguei para a empresa para receber a taxa de atraso demitida, mas eles me disseram que apenas davam um deles a cada 6 meses e eu já tinha um no xxxx xxxx. Fui informado de que a lei federal protegia os consumidores na Lei de Proteção ao Pagamento XXXX XXXX XXXX. Liguei para a empresa de volta com essas informações e me disseram que Bill foi apresentado apenas pelo Congresso, mas nunca ratificado em lei. Minha família está se esforçando para atender às nossas demandas com um orçamento limitado fixo para passar pela escola. Essa taxa de atraso foi uma surpresa inesperada para nossas finanças. Nós nos esforçamos para efetuar nosso pagamento a tempo, mas não tínhamos idéia de que o correio demore tanto ou que essas proteções não estão realmente em vigor. Por favor ajude!\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: A venda a descoberto foi aprovada pela Chase em xx/xx/xxxx. A liquidação foi agendada originalmente para xx/xx/xxxx. Os documentos foram enviados à Aprovação do HUD para o HUD em XX/XX/XXXX. Quanto mais perto não respondeu ou mesmo revisou os documentos até o dia anterior a liquidação (xx/xx/xxxx). Ao revisar as informações, o mais próximo aconselhou que não conseguiu obter a aprovação do HUD a tempo de liquidação e que foi apoiado em outras aprovações do HUD. Ele aconselhou que reagendamos as liquidação para o xx/xx/xxxx ou xx/xx/xxxx.   O assentamento foi remarcado para o XX/XX/XXXX, (hoje). Os documentos de aprovação do HUD foram reenviados na manhã de xx/xx/xxxx, dando três dias úteis completos para revisão e aprovação, conforme instruído na carta de aprovação de venda a descoberto. A aprovação do HUD ainda não foi emitida. Ao falar com o departamento de escalados de perseguição, não há nada que eles possam fazer e vários arquivos são backup. Fui aconselhado a cancelar as liquidação e aguardar uma resposta do departamento de venda a descoberto.   A carta de aprovação da venda a descoberto expira amanhã. Chase não está lidando com esse arquivo corretamente e vamos perder um comprador. Esta casa acabará em execução duma hipoteca.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta:  Alguém que pode ajudar\n",
            "======================================================\n",
            "Texto: Esta agência ainda tem a conta da minha esposa listada no meu arquivo de crédito. O suficiente é o suficiente xxxx que eles verificaram através da perseguição. Essa conta me pertence, que é uma informação falsa e prejudicou minha classificação de crédito. Por favor me ajude\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  a remover isso.\n",
            "\n",
            "A prime\n",
            "======================================================\n",
            "Texto: Esta é a minha segunda reclamação contra Chase. Meu original foi fechado, o caso # para referência é xxxx. Em sua resposta que anexei abaixo, eles afirmaram novamente que não estamos em comunicação com eles. Que como provamos nesse caso como falso. Antes de registrar a primeira reclamação, enviamos por e -mail, enviamos por correio e enviamos um fax (de um local de serviço do Chase Bank), toda a documentação que dizia que nunca fornecemos. Essa documentação também estava na queixa original que eles claramente não leram. Desde que o caso original foi fechado no XXXX, chamamos o número de telefone que eles forneceram na carta duas vezes por semana, toda semana. Ninguém jamais pegou, e ninguém jamais retornou nossa ligação. Essas são práticas enganosas e queremos falar com alguém imediatamente e reconhecer o recebimento da documentação que enviamos seis vezes (três vezes por e -mail, duas vezes por fax e uma vez por correio)\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  e verificar completamente noss\n",
            "======================================================\n",
            "Texto: XX/XX/XXXX XXXX XXXX XXXX Chase Auto Finance XXXX XXXX XXXX XXXX XXXX, New York XXXX Fax ( XXXX ) XXXX RE : CHASE AUTO FINANCE # # # # # # # # # # # # # # # Dear XXXX XXXX : I Realmente preciso de alguém para ouvir minha situação.   Estou escrevendo para você por desespero, pois não estou chegando a lugar algum com o atendimento ao cliente na Chase Auto Finance.   Eu falei com 6 indivíduos que começam com xxxx xxxx, em xx/xx/xxxx, xxxx em xx/xx/xxxx, xxxx xxxx, xxxx e finalmente xxxx id # xxxx tudo em xx/xx/xxxx. Foi explicado repetidamente para mim, muitas vezes, que você tem uma política que um pagamento não pode ser adiado, a menos que tenha havido um histórico de 9 pagamentos para perseguir o Auto Finance. Estou claro sobre suas regras para adiar o pagamento. Entendo que só pagou em um pagamento em xx/xx/xxxx e agora estou 11 dias atrasado pelo meu pagamento xx/xx/xxxx.   Deve haver em algum lugar da sua empresa que reconheça circunstâncias atenuantes que não fazem parte do seu livro de brincadeiras. Alguém precisa entender que minha situação é uma emergência. Como com nossos recentes furacões em Porto Rico, Texas e Hipotecas da Flórida não podem ser pagas a tempo devido a situações além do controle de hipotecas.   Estou em uma emergência que está além do meu controle, mas seu pessoal no atendimento ao cliente não pode ver além do livro de brincadeiras pelo qual eles operam.   Meu lindo carro foi roubado em xx/xx/xxxx. Comprei meu xxxx xxxx xxxx xxxx xxxx em xx/xx/xxxx, então eu só o tive em minha posse por 6 semanas. O ladrão que roubou meu carro levou um passeio de alegria com três agências policiais em perseguição em uma perseguição de 170 mph no Condado de Xxxx. O carro teve danos extensos e agora tem danos extensos do que se acreditava originalmente por causa da perseguição de alta velocidade.   O seguinte foi explicado a cada nível de atendimento ao cliente: entenda que, quando recebi meu empréstimo do Chase, eu era um xxxx xxxx no xxxx xxxx e tinha um espaço de trabalho xxxx xxxx xxxx. Mudei de trabalho nessas 6 semanas para trabalhar na área xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx porque é uma das melhores empresas XXXX do país e o pagamento é muito melhor. Mas como eu vivo a 75 milhas do xxxxxxxx xxxx, em xxxx, devo ter meu próprio transporte, pois os locais do evento nunca são os mesmos. Não posso levar o transporte público para cada novo local, onde um evento XXXX está sendo realizado, dado o transporte público e o transporte público não opera 24 horas por dia, 7 dias por semana, para me levar para casa nas primeiras horas da manhã. Por favor, entenda que não perdi meu emprego e quando posso marcar uma carona para aceitar um emprego, eu faço. Mas estou sofrendo muito durante esta temporada muito movimentada, perdendo muitos shows bem remunerados, porque não tenho meu próprio transporte. Sim, uma lição bem aprendida que vou adicionar aluguel de carros ao meu seguro no futuro. Mas isso é a visão posterior e não me ajuda no momento.   Fui informado pela instalação de autobody XXXX XXXX que levará 6 semanas para reparar meu carro para trazê -lo de volta à condição intocada quando comprava o carro. Obviamente, existe a dedutível {$ 500,00} para tirar meu carro do reparo do autobody. Mas no início do XXXX, quando vou recuperar meu carro, e ficarei 2 meses para trás nos meus pagamentos para perseguir as finanças automáticas e viver com medo de que meu carro seja recuperado.   Mas, como você está ciente, você tem o direito de recuperar meu carro se eu estiver um dia atrasado e é por isso que implorei para receber um pagamento diferido até xx/xx/xxxx como meu próximo pagamento data de vencimento.   Fui negado em cada novo nível solicitando adiamento, citando o histórico de pagamentos de 9 meses.   Posso verificar minha situação, pois tenho uma intimação para uma data do tribunal de xx/xx/xxxx para testemunhar contra a pessoa que roubou meu carro. My court case XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX, XXXX XXXX XXXX is the XXXX XXXX in the XXXX XXXX XXXX office and her phone number is ( XXXX ) XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX   Eu realmente preciso de alguém para me ouvir e me dar a data de vencimento do pagamento diferido que estou solicitando.   Por favor, seja essa pessoa. Este é o meu primeiro carro em meu nome e não quero xxxx no meu crédito com uma reintegração de posse. Entre em contato comigo o mais rápido possível com uma resposta positiva.   Agradecendo com antecedência,\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  XXXX XXXX\n",
            "======================================================\n",
            "Texto: Eu tinha conta no Chase Bank por quase alguns anos recentemente recebi uma carta que eles vão fechar minha conta que liguei para o papel para descobrir por que eles não me deram respostas que tentei abrir uma nova conta bancária, mas não posso e não Acho que é porque algo que eles escreveram na minha história e eu não sei por quê. Eu não escrevi nenhum cheques ruins ou fiz nada fraudulento, então o que está acontecendo?\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "\n",
            "É desconcertante que o Chase Bank\n",
            "======================================================\n",
            "Texto: Estou registrando uma queixa por um cartão de crédito que nunca recebi e nunca recebi declarações. Até que liguei, não sabia que o cartão de crédito era para xxxx. (Cartão de Chase # xxxx). Quando liguei e conversei com o Chase, eles me informaram que as acusações eram {$ 30,00} por mês para um cartão XXXX que nunca recebi nem usei. Eles não declaram nada ao solicitar o cartão de uma taxa mensal de {$ 30,00} e a taxa não é nada. Se o cartão fosse usado para fazer uma compra, eu pude entender, mas {$ 30,00} por mês para que o cartão seja simplesmente assalto pela Amazon. Eu gostaria de removê -lo se puder, se não, algo para mostrar que os pagamentos nunca se atrasaram, já que nunca recebi nenhum declarações ou um cartão de crédito deles. Para constar, depois de conversar com eles, paguei o cartão e o fechei.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Você pode me ajudar com ess\n",
            "======================================================\n",
            "Texto: Meu marido e eu estávamos programados para fechar nosso refinanciamento com o Chase Bank no xxxx / xxxx / 17. Xxxx xxxx foi o agente de assentamento e enviando xxxx xxxx, (notário, para a casa para concluir a assinatura do documento. Naquela noite. Meu marido e eu já tínhamos planos de jantar e não podíamos. Então foi concordado em assinar os documentos no dia seguinte em xxxx. Em xxxx, xxxx xxxx ainda não havia chegado, então falei com xxxx xxxx (um oficial de empréstimo com Chase que estava cobrindo xxxx xxxx o lo com o qual começamos. XXXX retornou minha chamada e declarou xxxx xxxx estava relatando que o fechamento ocorreu na sexta -feira, quando, na realidade Meu marido para adiar assinando os documentos. Após a discussão com Chase e xxxx xxxx sobre as datas nos documentos que não correspondem à data do fechamento, meu marido e eu concordamos em assinar os documentos com o xxxx data para resolver o fechamento. Os documentos foram assinados na presença de xxxx xxxx em xxxx em xxxx. Nesse ponto, o empréstimo estava programado para financiar o XXXX, a data em que nosso bloqueio expirou. Na quinta -feira, xxxx, recebi em chamado do XXXX XXXX, um gerente de encerramento de Chase dizendo que estava ligando para reagendar o fechamento desde que o notário destruiu os documentos. Após várias tentativas de encontrar um tempo comum, reagendamos o fechamento para segunda -feira, xxxx xxxx. Depois de pensar nisso por algumas horas, cancelamos o fechamento e insistimos que não usamos xxxx xxxx. Chase concordou em cobrir a extensão de bloqueio para garantir que nossa taxa permanecesse a mesma. XXXX tem sido o nosso principal contato durante a última parte dessa bagunça. Recebemos uma mensagem de um representante de vendas, xxxx xxxx, com xxxx xxxx na sexta -feira xxxx. Sua mensagem menciona os \"obstáculos\" que experimentamos com nossa refinanciamento. Também recebemos um cartão -presente {$ 50,00} xxxx xxxx para o nosso inconveniente. Na quarta -feira, xxxx xxxx que não tínhamos ouvido falar sobre por que ou como os documentos foram destruídos. Insistimos que o Chase iniciou um [processo de um monitoramento de proteção de crédito. No xxxx xxxx, recebemos um envelope xxxx de xxxx xxxx com as informações de monitoramento de crédito. Ainda não assinamos nada, já que não sabemos dessa empresa em particular. Ao saber que Havia um notário envolvido, em xxxx xxxx em xxxx, conversei com xxxx com o escritório xxxx xxxx xxxx que me encaminhou para xxxx xxxx no nível de estado. Ela não fez mais esse trabalho, mas pegou minhas informações e passou a xxxx (xxxx ) Xxxx. Conversei com xxxx na quarta -feira xxxx em torno de xxxx. Ele pegou a essência geral das informações e sugeri que eu entrei em contato com xxxx xxxx xxxx xxxx xxxx xxxx também tive dê dar n me o nome de alguém nesse escritório- xxxx xxxx. Deixei uma mensagem para ele na quarta -feira. Ele devolveu minha ligação com uma mensagem informando que transmitiu minhas informações para o departamento correto. Falei com xxxx xxxx na quinta -feira. Eu compartilhei verbalmente as informações sobre o nosso pesadelo de refinanciamento e ela pediu que eu concluísse o formulário on -line. Nesse ponto, não reagendamos o fechamento do refinanciamento. A única despesa fora do bolso que pagamos até este ponto é a taxa de inscrição {$ 500,00} para perseguir. Teremos que examinar a extensão da bloqueio de taxa e também se precisarmos comprar a taxa para chegar onde estávamos. Os custos restantes listados abaixo foram agendados para serem pagos no fechamento: $ xxxx - Taxa de processamento $ xxxx - Taxa de serviço tributário $ xxxx - Taxa de avaliação $ xxxx - Resumo/Pesquisa de título {$ 550.00} Seguro de título dos credores $ xxxx - Serviço de gravação {$ 310.00} Taxa de fechamento de liquidação Grand Total- {$ 2100.00} Meu marido conversou com um xxxx em xxxx xxxx no xxxx para obter o nome da companhia de seguros de título que o departamento jurídico do XXXX XXXX não divulgaria. Falamos com xxxx xxxx com o Escritório de Escalação com Chase. Ele foi convidado a compartilhar que Chase não estava disposto a fazer nada para corrigir a situação. Recobrimos o fechamento do nosso empréstimo para sexta -feira xxx em xxxx. A partir de hoje, xxxx em xxxx, não recebi um CD atualizado para este novo fechamento com uma nova empresa de título. Também solicitei uma cópia da carta de proteção de fechamento da Chase que eles receberam do XXXX XXXX, que não foi enviado a partir desse momento. Acredito que a Chase violou nossos direitos como consumidores por não financiar o empréstimo na data em que foi agendado. Não houve compensação pelo empréstimo não financiamento (e esperamos financiar até o final do XXXX XXXX Mês mais tarde do que o previsto).\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta:  Somos mais desconfortáveis\n",
            "======================================================\n",
            "Texto: Eu tenho um novo cartão de crédito. Neste cartão, recebo uma carga xxxx em uma estação XXXX no TN. Na data em que postou, eu não estava perto desse estado. Enviei minha disputa neste cartão de crédito para perseguir. Depois de três meses, eles me informam que minha disputa é negada porque disseram que uma pessoa tinha o cartão de crédito fisicamente lá e o usou. Tenho várias pessoas que podem testemunhar que não estava perto desse estado nessa data. O que eu faço sobre eles ainda tentando me cobrar?\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "Primeiro, você precisa desc\n",
            "======================================================\n",
            "Texto: Chase Mortgage arruinou minha vida. Olha como eles relataram ... xxxx mentira! Se eu não tivesse sido tão inadimplente, você não acha que eles foram excluídos? Eu paguei a trapaça de trapaça inflada sem fundamento, nota mensal por 16 anos desde 2001. Eles disseram que pensavam que tínhamos xxxx anos, então eles calcularam mal o pagamento! Eles são criminosos internacionais -o que xxxx, ou xxxx e supostamente compensam os titulares de hipotecas. Não eu - eles sempre dizem que estou atrás \"... Uh, por causa do garantia, ou uh, por escassez no último pagamento composto por mentirosos blá blá ladrões. Além de duas vezes, eles coletaram {$ 12000,00} em dinheiro em xxxx ou eles foram excluídos !!! e depois venderam minha casa de qualquer maneira, mas eu estava na venda e a parou! Eles estão mentindo sobre tudo. Outra tentativa, seus advogados tiveram que Tenha verificação certificada para {$ 12000.00} em xx/xx/xxxx \"ou então ''. Eles negam sistematicamente meu pedido de harpa ou juros hipotecários reduzindo '' Os subscritores rejeitaram seu aplicativo '', eles garantem que algo dê errado para que eu não se qualifique ... como o relatório de crédito. Agora eu vejo o dano que eles continuam a fazer. Paguei por essa casa, que, a propósito, era um XXXX original em 2001 de {$ 150000.00}. Este relatório é um refinanciamento de 7,5 para 5,5. Então eles disseram que éramos xxxx !!! O valor então financiado não era {$ 150,00}, ooo. Era {$ 120,00}. Talvez alguém agora possa investigar sua ruína de mim. Eles ligaram para meus credores - eu sempre fazia meus pagamentos, mas, de repente, todos os fornecedores cortaram meu crédito. Fiquei atordoado. Chase fez isso. Além disso, eles dizem que eu estava atrasado o tempo todo ... ninguém pode ver o que está acontecendo? Além disso, finalmente - na XXXX, meu marido recebeu muito pagamento da OT. Em seguida, em xx/xx/xxxx ou xx/xx/xxxx, xxxx o interrompeu. Ele ainda trabalha muitos eventos, mas fica em casa 2 ou 3 dias por semana, quando isso acontece, então trabalha 18 horas dias. Perdemos US $ xxxx todos os meses. Além disso, tivemos um bebê, e ele nunca cuidaria - para o médico, etc. Ela ficou doente mais do que o diretor queria e fui demitido !!! em xxxx. Encontrei um ótimo trabalho, mas apenas 20 horas. Ainda não há 40 horas de emprego. Então, fomos de tudo bem, com poupança adicionada todos os meses (!) A nenhum salário de mim + então xxxx, e nenhum OT dele. Chase Mortgage naquele mês NOTA DE {$ 1000,00} a {$ 1700,00} e mantido por 6 meses. Eles nos quebraram de propósito e eu perdi meu bom nome de crédito.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Odeio Chase Mortgage !!! Estou tent\n",
            "======================================================\n",
            "Texto: Abri uma conta corrente pessoal, a conta ficou aberta por apenas três dias antes de ter sido suspeitamente fechada. O banco não garantiu a segurança da minha conta ou seguiu o protocolo de atendimento ao cliente.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "\n",
            "Neste caso, o primeiro\n",
            "======================================================\n",
            "Texto: Solicitei meu valor de pagamento da hipoteca em xxxx/xxxx/xxxx e me disseram que eu deveria receber o valor da recompensa em 3-5 dias. Liguei novamente em xxxx/xxxx/xxxx e me disseram que a cotação de pagamento não foi gerada e que uma solicitação foi feita em xxxx/xxxx/xxxx. Liguei para o banco novamente em xxxx xxxx, xxxx. Eles me disseram que minha cotação de pagamento foi enviada por correio no xxxx xxxx, xxxx, e eu deveria receber cotação de pagamento em xxxx xxxx ou xxxx xxxx. A partir de hoje, xxxx xxxx, xxxx, ele não foi recebido. Pedi que a cotação de pagamento fosse dada verbalmente por telefone e fui negado o pedido.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Eles disseram que eu teria\n",
            "======================================================\n",
            "Texto: Entrei em contato com o Chase em xx/xx/xxxx, xxxx, xxxx e xxxx em relação à remoção do PMI. Eu estava com a impressão de que, se eu pagasse até 80 % do meu valor de hipoteca original, não precisaria de um BPO. Ao falar com os gerentes de escalada da Chase, me disseram que precisaria de um BPO, independentemente de quanto dinheiro paguei em relação ao meu diretor e independentemente de pagar mais de 20 %. Em seguida, recebi um aviso por escrito afirmando que não precisaria obter um BPO se pague a 80 % do meu valor de empréstimo original. Liguei para Chase novamente e fui informado pela primeira vez hoje que Chase homenagearia um acordo por escrito, mas que eles não administram meu PMI e que eu teria que entrar em contato com o negócio que lida com meu PMI. Infelizmente, o Chase não pôde me fornecer o nome da empresa ou o número de telefone para entrar em contato com o meu PMI. Chase afirmou, pois eu precisaria de um BPO, independentemente do dinheiro pago (mesmo que fosse mais de 80 % do valor original). A Chase afirmou que eles reembolsariam meus pagamentos principais extras. Estou relatando isso devido a informações incongruentes e contraditórias, que não estão alinhadas com os requisitos do PMI.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Atualmente, minha conta corrente possui um bloco ou restrição impedindo o uso de depósito móvel, isso se deve ao número de vezes o excesso de retagonização/atualmente exagerado. Eu não tenho ramos e / ou caixas eletrônicos localizados perto de mim em Connecticut, precisarei usar depósito móvel para trazer a conta positiva e para futuros depósitos. O representante com quem conversei hoje declarou hoje que poderia transferir fundos da economia, no entanto, um regulamento federal impede uma série de transferências XXXX por ciclo de declaração. Escusado será dizer que isso não resolve a matéria.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "\n",
            "Eu sugiro o segu\n",
            "======================================================\n",
            "Texto: Eu tenho uma placa CHASE XXXX XXXX, anunciada como um cartão de viagem. Eu tenho esse cartão há ~ 4 anos e usei extensivamente o cartão. Continuo em boa posição com o cartão, pagando as quotas a tempo. Em xx/xx/xxxx, tentei fazer uma reserva para um voo em xxxx por {$ 210,00}, mas Chase rejeitou essa transação. No momento, tenho mais de {$ 15000,00} no limite de crédito disponível. Liguei para Chase e eles declararam que trancaram minha conta, mas se recusaram a explicar por que a conta estava bloqueada e me pediram para ligar para outro dia. Eles também me enviaram uma mensagem de texto solicitando -me para confirmar que a transação era realmente minha. Respondi imediatamente confirmando que a transação era realmente minha. A transação, no entanto, continuou a ser rejeitada. Sou um viajante ávido e um XXXX, esse comportamento é claramente transações de segmentação xenófobas em uma região específica e não estão alinhadas com as expectativas de um cartão de viagem.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Gostaria de entender por que a trans\n",
            "======================================================\n",
            "Texto: Eu tenho um empréstimo com o Chase Auto. Eu sempre fiz meus pagamentos a tempo. Por alguma razão, percebi que havia um pagamento atrasado no meu relatório de crédito. Como você pode ver, eu sempre tive um registro de pagamento estelar com esta empresa. Tentei entrar em contato com o XXXX, XXXX e o Chase Auto sem resolução bem -sucedida. XXXX e XXXX estavam me relatando tarde. Definitivamente, houve um erro da parte deles. Eu nunca estive 30 dias atrasado xx/xx/xxxx, xx/xx/xxxx, xx/xx/xxxx-xx/xx/xxxx, xx/xx/xxxx-xx/xx/xxxx, xx/xx/xxxxxx/xxx/xx/xxxx, xx/xx/xxxxxx/ Xx/xxxx.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "Agora estou tentando contatar\n",
            "======================================================\n",
            "Texto: Meu nome é xxxx xxxx e tenho uma reivindicação de sistemas de verificação do JP Morgan Chase Bank no meu relatório de sistemas de cheques para uma conta corrente antiga que tive com eles em 2010, que foi composta devido a roubo de identidade. Eles me enviaram XXXX DISC este ano para perdoar a quantidade que eles foram atraídos, mas ainda me têm em sistemas de cheques por fraude. Alguém roubou meu cartão de débito e tentou depositar cheques de cartão de crédito roubados em minha conta, fiz um relatório policial e eles ainda fecharam minha conta e agora não abro uma conta em nenhum lugar. Pleas ajudam com isso obrigado\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: . \n",
            "\n",
            "Lamentamos ouv\n",
            "======================================================\n",
            "Texto: Logado no Chase para pagar minha hipoteca em xx/xx/2016. Eu fiz o pagamento. Parecia ter passado. Um mês depois, recebi um aviso de atraso por correio. Chase não me ligou ou enviou uma carta de lembrete. Agora eu tenho uma taxa tardia {$ 59,00} e um aviso de atraso no meu crédito. Imediatamente ficou on -line para pagar depois de receber o aviso tardio, registrado no pagamento. Mais uma vez, não levou. Chamou Chase Direct e pagou dessa maneira. Nunca tivemos nenhum tipo de problema quando o XXXX XXXX teve o empréstimo pelos 4 anos anteriores, antes de Chase assumir o controle. Parece que Chase está preparando os mutuários para o fracasso, para que possam ganhar dinheiro extra com as taxas atrasadas.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Chase Bank xxxx xxxx xxxx xxxx xxxx mi xxxx seu departamento A.C.H pressionou o último valor da verificação do SSID da minha esposa {$ 1200,00} Isso aconteceu em xx/xx/2019. Esse cheque entrou em nossa conta no XX/XX/2019 naquela época, o banco mantinha os fundos por dois dias e depois o lançou. Foi verificado por três representantes diferentes em nossa previdência social que esse cheque foi um bom cheque e os três não entendem por que eles estão segurando o dinheiro. Na Chase XXXX, me disse que o SSID estava mantendo o dinheiro no dia do espera xxxx de xxxx um problema é que não há número de telefone para ligar para ajudar a resolver problemas com o departamento de perseguições da ACH. Toda a comunicação é feita por e-mail. O SSID não estava segurando a perseguição de dinheiro. Seu departamento de ACH, nós nos destacamos no XXXX, está revelando uma carta da SSID que eles não recuperarão esse dinheiro e queriam que o documento envie por fax para eles afirmando isso. O problema é que o SSID não escreve cartas e não me ajudaria com uma carta. Por isso, fui instruído pelo SSID de que essas etapas são a única maneira de obter o meu {$ 1200,00} de folga. # 1 Cubra o déficit de {$ 330,00} # 2 tem que enviar o dinheiro de volta ao SSID # 3 Eu tenho que registrar um formulário de três páginas para solicitar meu próprio dinheiro de volta e esperar até 60 dias, cobri o valor apropriado OM na segunda -feira O XXXX, hoje a partir do minuto, o dinheiro ainda está em espera. Eu comuniquei tudo isso com o Chase Bank (xxxx) no xxxx. Aqui está o que parece para mim 1ª temporada de festas sem minha esposa, eu xxxx em xxxx para cuidar dela com xxxx xxxx, estou em uma pequena renda do SSI que tive que inventar {$ 1500,00} para cobrir o cheque para Faça o Chase enviar o dinheiro de volta e ter dinheiro para pagar contas. Este é o dia 12 com o meu dinheiro em espera e ninguém no Chase parece se importar\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: . O que mais posso fazer?\n",
            "======================================================\n",
            "Texto: O JP Morgan Chase Bank continua ligando para o meu marido e minha casa. Queremos notificação por escrito sobre esta dívida. Queremos que essas chamadas cessassem.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "\n",
            "Lamentamos muito o\n",
            "======================================================\n",
            "Texto: Eu tive um cartão Amazon.com através dos Serviços do Card Card. XX/XX/XXXX, me inscrevi para um programa chamado um programa de liquidação de equilíbrio. Este é um programa que me dá uma taxa de juros de 2 %, desde que eu faça pagamentos mensais pontuais. Os termos do programa estipulam que, se eu perder os pagamentos xxxx seguidos, a conta será cobrada. Fiz meu primeiro pagamento bem-sucedido xx/xx/xxxx e configurei pagamentos recorrentes para não precisar monitorar os pagamentos todos os meses, pois sou uma nova mãe que trabalha em tempo integral e tem muito no meu prato. Xx/xx/xxxx, recebi uma ligação da Chase dizendo que meu pagamento foi devolvido a eles. Verifiquei o número da minha conta com eles e eles haviam inserido incorretamente as informações da minha conta, fazendo com que o pagamento fosse devolvido. Fiz um pagamento por telefone naquele momento e, novamente, configurei pagamentos recorrentes com o representante. Novamente xx/xx/xxxx, recebi uma ligação afirmando que meu pagamento foi devolvido. Mais uma vez, paguei ali por telefone e verifiquei o número da minha conta novamente com eles. Eu tinha certeza de que o sistema deles foi atualizado corretamente e isso não aconteceria novamente. XX/XX/XXXX, recebi uma chamada de um representante do Chase, informando que os pagamentos XX/XX/XXXX e XX/XX/XXXX foram devolvidos e que eu precisava fazer um pagamento imediatamente para evitar uma carga. Eu estava obviamente frustrado, já que isso já havia acontecido várias vezes, e eu me ofereci, novamente, pagar por telefone naquele momento. O representante me disse para ir a uma filial do Chase e fazer um pagamento em vez de pagar por telefone. Perguntei -lhe várias vezes: \"Você tem certeza de que não apenas pagou por telefone? '' Ele me garantiu que seria melhor entrar em uma filial e pagar. Eu fiz isso, mas a conta já havia sido cobrada Fora. Quando liguei para Chase para reclamar, disse a um supervisor que me disseram um representante para entrar em uma filial e fazer um pagamento. Ela disse que ele não deveria ter dito isso e que eles fariam uma revisão de chamada para ver se foi o que foi dito. Ela me disse que, se ele dissesse isso, eles corrigiriam. Ela me disse que me ligaria pessoalmente quando a revisão da chamada tivesse feita, mas eu nunca mais ouvi dela. A próxima chamada Recebi de Chase de outro representante tentando coletar dinheiro na minha conta cobrada. Conversei com vários outros supervisores e eles disseram que só podem reverter a acusação que está no meu relatório de crédito se foi um erro bancário. Eu tenho uma carta da minha união de crédito afirmando que eles nunca receberam nenhum pedido de pagamento da perseguição Xx/xx/xxxx ou xx/xx/xxxx e que eu tinha fundos suficientes em minha conta durante esse período para fazer meu pagamento de {$ 130,00}. Perguntei a Chase se posso ouvir minhas ligações que foram gravadas para mostrar que eu lhes dei as informações bancárias corretas, mas eles me disseram que eu precisaria de uma intimação para fazer isso. É claramente o erro deles, se eles conseguiram deduzir com sucesso os fundos da minha conta por telefone, mas nunca enviou pedidos de pagamento ao meu banco para me manter inscrito no programa. Tenho evidências corroboradas do meu banco, afirmando que eles nunca receberam qualquer pedido de pagamento do Chase XX/XX/XXXX ou XX/XX/XXXX e que eu tinha fundos suficientes em minha conta.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Como posso resolver esse problema\n",
            "======================================================\n",
            "Texto: No XXXX, o Chase de repente feche todos os meus cartões de crédito porque gasto mais de 10k no dia xxxx xxxx usando meu cartão de assinatura xxxx xxxx. Eu nunca tenho um atraso no pagamento e minha pontuação de crédito nunca abaixo do xxxx. Liguei para o número no cartão. O representante me disse que já tomou a decisão e não fará nenhuma reconsideração. Isso é injusto para o cliente e traga muitos problemas para mim.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Eu entendo completamente\n",
            "======================================================\n",
            "Texto: Agradecemos que você reserve um tempo no seu dia ocupado para nos aconselhar sobre esse pequeno detalhe. Em 2013, pagamos nosso saldo de cartão de crédito (menor que {$ 100,00}) 30 dias atrasado. Agora, a partir de quatro anos depois - esse atraso no pagamento permanece como o único pagamento atrasado em todo o nosso histórico de crédito, abrangendo décadas e 100s de credores. O impacto negativo desse incidente singular e isolado em nosso crédito, em nosso custo de seguro, em nosso custo de empréstimos - vai muito além de qualquer penalidade razoável por uma infração tão menor. Além disso, o relato contínuo desta questão não contribui nada para o objetivo pretendido de tais relatórios, ou seja, exposição de maior risco associado ao uso de crédito. Fizemos várias tentativas ao longo desses anos para solicitar assistência da Chase - a cada vez, recebendo uma razão diferente e propositalmente ambígua de sua incapacidade de fazer sentido com esse não sentido. Como você tem tempo - você gentilmente forneceria orientação sobre como podemos resolver melhor esse problema? Chase Executive Office XXXX Oh xxxx Stop OH XXXX Obrigado, xxxx\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Conta # xxxx no xxxx O xxxx Eu fui atingido com duas taxas de rascunho. Meu saldo atualmente xxxx, xxxx das taxas. Consistindo em {$ 34.00} cada. Entrei em contato com o XXXX e fui informado de que as taxas não podiam ser dispensadas manualmente, mas ele poderia enviar uma revisão e elas poderiam ser revertidas. O agente também afirmou que eu poderia retornar dentro de 24 horas para verificar o status. Na mesma noite em que liguei de volta e peço para falar com um gerente, a manjedoura passa a me informar que eu tinha que esperar xxxx a chance de ser revertido. No xxxx xxxx, xxxx, eu chamo de manhã para verificar o status, o representante e o gerente me informam que este não é um processo ou política da empresa. Fui enganado por um representante e um gerente.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Estou indagando se essa taxa\n",
            "======================================================\n",
            "Texto: De volta a xx/xx/xxxx, eu estava no xxxx xxxx tentando comprar um carro. Concluí um pedido de empréstimo e dei autorização xxxx para puxar meu relatório de crédito. Eu não sabia que xxxx sem meu consentimento enviou a inscrição em xx/xx/xxxx a outros quatro bancos na esperança de que um deles aprove meu empréstimo e, no processo, todos os bancos puxaram meu relatório de crédito, fazendo -me que tivesse um Inquérito de sucesso e afetando negativamente meu relatório de crédito. Para muitas consultas, igual a classificação de crédito negativo e permanecerá no meu relatório por dois anos. Entrei em contato hoje, xx/xx/xxxx e fui transferido para a unidade do departamento de crédito e fui informado por xxxx que, porque xxxx enviou ao aplicativo eletronicamente que seu procedimento deve fazer uma verificação de crédito, mesmo que eles não tenham escrito consentimento de eu para executar uma verificação de crédito. XXXX se recusou a me fornecer qualquer informação para que eu possa enviar uma disputa por escrito e solicitar uma carta de exclusão. Por XXXX Chase, não solicitará que a consulta seja excluída.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Vou enviar cartas para\n",
            "======================================================\n",
            "Texto: XXXX a.k.a. XXXX XXXX XXXX XXXX XXXX ( XXXX ) a.k.a. XXXX XXXX XXXX a.k.a. XXXX XXXX XXXX a.k.a. XXXX XXXX a.k.a. Bank One complaint with Consumer Financial Protection Bureau ( CFPB ) XXXX/XXXX/XXXX - Original debt to XXXX discharged in bankruptcy ( please note that Todos os números de conta fornecidos para mim são diferentes do número da conta original de xxxx) xxxx/xxxx/xxxx - xxxx xxxx xxxx xxxx xxxx xxxx xxxx xxxx enviado letra de coleta para {$ 9400.00}; XXXX XXXX XXXX XXXX, XXXX, XXXX XXXX, phone : XXXX, fax : XXXX XXXX/XXXX/XXXX - XXXX permanently closed account per A. XXXX XXXX/XXXX/XXXX - XXXX XXXX XXXX sent collection letter for {$10000.00} ; Xxxx xxxx. Xxxx xxxx, xxxx, xxxx xxxx. XXXX Conta permanentemente fechada xxxx/xxxx/xxxx - arquivei reclamação on -line contra eles.   Xxxx/xxxx/xxxx - xxxx xxxx xxxx xxxx chamado meu telefone celular (nunca foi listado com dívida original) procurando meu cônjuge, afirmou estar representando xxxx xxxx xxxx e ameaçou ação legal se $ xxxx não foi pago sobre o telefone, insistido que insistiu que insistiu que insistiu que insistiu que insistiu que insistiu, insistiu que insistiu, insistiu que insistiu, insistiu que insistiu, insistiu que insistiu, insistiu que insistiu, insistiu que insistiu, insistiu que insistiu, insistiu que insistiu, insistido por telefone, se a dívida não foi paga xxxx xxxx e ameaçou uma ação legal. A dívida foi feita em xxxx xxxx e os pagamentos foram feitos até xxxx, recusou -se a enviar qualquer documentação de suporte, xxxx xxxx/xxxx/xxxx - contatou o advogado de falência e foi informado de que era uma farsa, não legal, e eu fui coberto pelas leis de falência.   Xxxx/xxxx/xxxx - xxxx xxxx xxxx disse que a dívida por {$ 4500.00} foi carregada em xxxx xxxx e depois vendida para xxxx xxxx em xxxx xxxx. Telefone: xxxx, linha de fraude: xxxx, atendimento ao cliente: xxxx xxxx/xxxx/xxxx - xxxx As coleções não possuem registros de dívida. Telefone: xxxx, xxxx xxxx/xxxx/xxxx - xxxx xxxx xxxx xxxx chamado meu telefone celular, procurando meu cônjuge, e queria coletar dívidas para xxxx reivindicando o mesmo que xxxx xxxx xxxx xxxx (mesmo folga?), Concordou que aceite a aceitação de xxxx xxxxxxxxxxxx (mesmo scam?), Accessou para aceitar a prova de xxxx xxxxxxxxxx xxx (mesmo vergonha?), Accessou a aceitar a prova de xxxx xxxxxxxxxx xxx (mesmo vergonha?), Accessou a aceitar a prova de xxxx xxxxxxxxxx xxx (mesmo scam?), Accessou a aceitar a aceitação de xxxx xxxxxxxxxx xxx. falência via fax ou email; Após o recebimento, preciso ligar para eles para verificar se o arquivo está fechado permanentemente, telefone: xxxx, xxxx, fax: xxxx, email: xxxxxxxxxxxxx xxxx/xxxx/xxxx - reclamação de arquivamento contra todos os envolvidos para encerrar esse assédio ilegal por dívida não existente .\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Eu tinha minha conta bancária fechada em xxxx no Chase em xxxx xxxx em xxxx xxxx xxxx xxxx xxxx, xxxx xxxx, ca xxxx, ca. Eu disse ao banco que teria fechado minha conta devido a fundos que não estão disponíveis também porque tenho um bloco de folha de pagamento; na minha folha de pagamento. Comecei a ouvir sobre os incêndios florestais em CA, perto de onde estava apostando, li na supervisão da Califórnia DBO; Esses condados precisam fechar suas contas, com o banco. Espero um tempo para ver se o banqueiro teria me dito e nunca o fiz, então encerrei minha conta, em boas condições com o banqueiro, naquele momento, perguntei se poderia voltar ao banco depois de fechar a conta, e o banqueiro declarou; sim.   Xx/xx/xxxx; Fui abrir uma conta segura de perseguição e estava bem, a conta foi aberta e não recebi meu cartão de débito pelo correio. Liguei para as mensagens de esquerda, e fui ao banco localizado no xxxx xxxx xxxx xxxx xxxx, xxxx xxxx, xxxx xxxx, ca xxxx. Disseram -me que eles estão fechando minha conta e não sabem o motivo.   Liguei para o número de atendimento ao cliente da Chase para descobrir o porquê, ela afirmou que não há razão, é a decisão dos bancos, e eu disse que a dose que significa ter um banco em Chase novamente. O representante de atendimento ao cliente, disse que sim.   Xx/xx/xxxx, enviei uma mensagem do Twitter para o suporte à busca. Xx/xx/xxxx afirmando que lamentam e podem me encontrar no sistema, respondo com o número da minha conta; Xx/xx/xxxx hoje, recebi uma mensagem do Twitter para ligar para Chase, liguei e não consegui criar uma conta\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: . O representante de atendimento a\n",
            "======================================================\n",
            "Texto: Em xx/xx/2016, meu marido dirigiu o xxxx do meu filho para xxxx em xxxx, NC. Eu o segui para que pudéssemos deixar o carro lá. Depois de receber uma ligação de XXXX, o proprietário, de que o carro estava pronto, fomos para lá, pagamos a conta e saímos. Nos dias seguintes, o carro começou a correr quase tão ruim quanto quando o deixamos inicialmente o deixamos. Meu marido e eu fomos ao XXXX pedir ao xxxx xxxx sobre os reparos. Foi quando os problemas começaram. Concordamos que os reparos do radiador foram concluídos satisfatórios, mas houve problemas consistentes com o desempenho ruim do carro. Desde nossa reivindicação inicial com xxxx e perseguição, consistentemente nos foi negado qualquer forma de resolução. Chase finalmente indicou que precisávamos de uma avaliação de terceiros. Meu marido entrou em contato com xxxx outras instalações de reparo, mas depois de receber a carta de negação final de Chase, meu marido ligou para as instalações de reparo XXXX e ninguém indicou que Chase havia feito alguma forma de investigação. Meu marido pediu que Chase nos fornecesse documentação por escrito, explicando suas descobertas sobre por que eles negaram minha reivindicação. Meu marido finalmente conversou com XXXX, uma mulher de Chase. Ela negou qualquer informação adicional e afirmou que, desde que eu recebi a carta de negação final, ela não discutia mais a questão e se recusava a fornecer qualquer comunicação verbal ou documentação por escrito, explicando suas descobertas sobre o motivo de negarem minha reivindicação. Eu forneci documentação adicional com esta reivindicação. Chase não me deu ajuda, o consumidor, mas prefere defender o negócio e não o cliente. Além disso, XXXX alegou que não tínhamos contatado nele sobre esse problema após o reparo inicial e ele também disse isso para perseguir. Incluí uma cópia do recibo certificado do XXXX, onde minha carta certificada para XXXX foi recebida.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Eu acredito que fui trat\n",
            "======================================================\n",
            "Texto: Eu solicitei uma oferta do xxxx xxxx xxxx e fui negado pelo Chase I já tenho uma conta de crédito xxxx e sou um titular de conta xxxx xxxx. Não fui recusado por esses motivos, em vez disso, eles me disseram que eu tinha deliquidades ou registros públicos. Eu tenho uma falência que estará fora do meu registro deste XXXX após 10 anos. Também me disse o número de novos pedidos, o que foi um problema com o Credit Bureau no ano passado que foi resolvido. Período de tempo desde o crédito mais antigo. Meu primeiro crédito foi quando eu era xxxx e agora sou xxxx e o último período de tempo desde o mais recente registro público. Recebi um empréstimo VA em xx/xx/xxxx e o último registro foi a falência em xx/xx/xxxx.   Quando arquivei a falência, listei Chase como credor e acredito que esse é o motivo desse declínio. Discriminação.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:   \n",
            "O CHASE nunca me disse as\n",
            "======================================================\n",
            "Texto: Eu agendei um pagamento para o meu cartão de crédito Chase em xx/xx/xxxx. O próximo pagamento foi devido em xx/xx/xxxx. Consulte minha declaração anexa e impressões de tela. Em xx/xx/xxxx, recebi uma notificação de que a conta estava vencida. Então minha esposa chamou Chase. Minha esposa falou com um representante chamado xxxx em torno de xxxx em xx/xx/xxxx. Ele afirmou que, embora as contas da conta em xx/xx/xxxx, a instrução não seja enviada por correio até xx/xx/xxxx. Portanto, o pagamento foi processado pelo mês anterior. Ele renunciou à taxa de atraso e confirmou que nossa taxa de juros de 0 % ainda era boa até xx/xx/xxxx. A principal razão pela qual estou apresentando a reclamação é a data da declaração. Seu sistema mostra claramente a data de cobrança como xx/xx/xxxx e a instrução também reflete xx/xx/xxxx. O histórico da declaração também mostra que sai no XX/XX/XXXX. Portanto, quando um cliente está agendando um pagamento, ele assumiria que o pagamento seria aplicado ao extrato atual se a data paga coincidir com a data da fatura e seu sistema on -line. Eu não deveria ter precisado fazer um segundo pagamento no mesmo mês para impedir que ele pareça que estava vencido. Indo com uma data de correspondência, xx/xx/xxxx, em vez do que é impresso na fatura e listado em seu sistema on -line, parece uma maneira bastante arcaica e uma maneira incorreta de processar pagamentos, especificamente porque os clientes dependem de dados on -line para agendar pagamentos em contas. Estou certo de que muitos clientes receberam taxas atrasadas por esse motivo e precisaram fazer dois pagamentos em um mês para corrigir a situação. Estou anexando a declaração e as capturas de tela da conta para seus registros.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Estou escrevendo que, devido à não compensação do CRA [xxxx], eles não conseguiram remover, atualizando as informações que eram inactUare e relatórios incompletos que foram desafiados no passe, acredito firmemente que eles estão violando a FCRA e são totalmente recusando -se a defender a lei\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: . \n",
            "\n",
            "Estou profundamente\n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, fiquei on -line para pagar xxxx xxxx xxxx xxxx xxxx, xxxx. Não ficou claro para mim que o pagamento feito foi pelo serviço futuro de XX/XX/XXXX a XX/XX/2017. Eu usei um cartão de débito do Chase Visa para pagar {$ 100,00}. Cancelamos nosso serviço do XXXX XXXX, pois vendemos nossa casa e estamos saindo do estado. Este pagamento é por um período pelo qual o novo proprietário é responsável. Ao entrar em contato com o Chase Bank, me disseram que eles não podem reverter a cobrança e que preciso buscar a restituição do fornecedor diretamente. Isso não é tempo eficaz.   Além disso, cancelei meu serviço xxxx xxxx na casa antiga e disse implicitamente ao xxxx xxxx que não os autorizo ​​a receber uma \"taxa de cancelamento\" do meu cartão de débito ou qualquer cartão de crédito ou diretamente da minha conta bancária, como Estou usando um serviço xxxx xxxx xxxx em nossa nova casa. Eles pertencem à mesma empresa. Além disso, documentamos evidências de que o serviço xxxx xxxx e o satélite interferiram no sinal para o nosso xxxx xxxx para nossos cães, causando nossos cães estar em perigo de fugir ou ser atropelado por um carro.   O Chase Bank tem sido terrível de lidar e tem sido tão inútil na resolução dessas situações.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Eles são lentos para responder,\n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, notei e arquivei 4 disputas de cobrança não autorizadas no Chase Bank. Eu fui vítima de fraude bancária. Alguém pegou minha conta bancária e a usou para fazer compras com xxxx. A primeira transação foi para {$ 1,00} em xx/xx/xxxx. A segunda transação foi para {$ 2500,00} em xx/xx/xxxx. A terceira transação foi para {$ 2,00} em xx/xx/xxxx e a transação Forth foi {$ 2100.00}. Em xx/xx/xxxx, percebi um erro e liguei para o Chase Bank para retirar a primeira disputa para {$ 1,00}. Após a apresentação da reivindicação, o Chase Bank emitiu um crédito temporário apenas para as duas primeiras transações no valor de {$ 2500,00}. Eu disse a eles que deveriam ter creditado minha conta por todas as transações porque a data da primeira transação agora era xx/xx/xxxx. Desde que relatei as compras não autorizadas dentro de 60 dias após a ocorrência, estou protegido sob o Regulamento E. Chase Bank precisa creditar minha conta pelas três transações no valor de {$ 4600,00}.   Entrei em contato com o XXXX e eles disseram que, como esses valores eram fraudulentos, o banco só precisa cancelar as transações da ACH. Eles não vão contestar o processo. Sem a aprovação do banco, eles não podem enviar os fundos de volta à minha conta bancária. Caso contrário, eles fariam.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "Como eu fiz disput\n",
            "======================================================\n",
            "Texto: Em xx/xx/2020. Fui online para o xxxx xxxx xxxx. Como anunciado, pedi um xxxx xxxx azul e vermelho para o meu neto que seu aniversário XXXX estava chegando. O preço do xxxx xxxx era {$ 240,00} mais eu paguei {$ 15,00} extra para acelerar o envio (5-8 dias) desde que o aniversário dos meus netos foi em 12 dias. 3 dias depois, xx/xx/2020 Meu filho me diz que eu deveria ligar para o JPMorgan Chase (meu banco) e cancelar a transação, pois a loja é uma farsa. Ligo para o JPMorgan Chase imediatamente e expliquei a situação. O dinheiro foi colocado de volta na minha conta. Em xx/xx/2020, 31 dias depois, recebi um pequeno pacote de xxxx. Quando abri, era uma alça de jogo para celular. Não é um xxxx xxxx. Eu ainda tenho isso. Um mês depois, xx/xx/2020 enquanto olha para o meu extrato bancário, percebo que o JPMorgan Chase reverteu a reivindicação. Nunca fui notificado sobre outras ações a serem tomadas sobre esse assunto. Liguei para o departamento de disputa de Chase e expliquei a eles o que havia recebido. Eles me disseram para enviar todas as provas por fax. Em xx/xx/2020, envio 2 fotos, 1 da figura mostra o que recebi e a segunda imagem mostra o que é um xxxx xxxx. Também páginas mostrando os e -mails da XXXX XXXX Company. Em xxxx xxxx, recebi uma carta informando que a transação foi autorizada e que minha reivindicação foi fechada. Eles citaram em nome da empresa. Eles também me disseram que ligaram para a empresa e a empresa disseram que enviaram o XXXX XXXX. Obviamente, a pessoa que fez a pesquisa não sabe a diferença entre um xxxx xxxx e uma alça de jogo móvel, que foi o que recebi. Estou com nojo de quão injusto fui tratado pelo banco. Eles não olharam para minhas evidências. Se esse fosse o caso, eles citariam em meu nome. Eu tentei entrar em contato com a empresa (xxxx xxxx xxxx, mas os e -mails foram enviados de volta para mim. Se eles fizeram uma pesquisa sobre essa empresa, verão uma farsa.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Qualquer um que olhou para os produt\n",
            "======================================================\n",
            "Texto: O Chase Bank me vendeu uma viagem de 12 dias a xxxx para pontos xxxx (no valor {$ 10000,00}), agora é impossível usar a viagem de acordo com o XXXX XXXX XXXX XXXX, você deve estar isoladamente por 14 dias ao entrar no país e \" Um estrangeiro que sai do país antes do final do isolamento sem aprovação especial, viola as leis do Estado de XXXX e sua próxima entrada não será aprovada. '' '(HTTPS: XXXX).   Então, basicamente, seria ilegal fazer a viagem, no entanto, o Chase Bank não me devolverá meus pontos.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "Gostaria de saber qual é a\n",
            "======================================================\n",
            "Texto: XXXX usou meu cartão de crédito para pagar a conta anual da minha esposa com a AARP Trusted ID. Eu não autorizei isso, mas eles usaram meu cartão. Um cartão que eu não sabia que tinha. Eles deveriam ter usado o cartão dela (o xxxx que ela os autorizou a usar) Eu nunca os autorizei a usar meu cartão. É a conta dela e ela paga por isso com o cartão. Isso constitui fraude, tanto quanto estou preocupado.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Recomendo que entre em cont\n",
            "======================================================\n",
            "Texto: Eu tenho minha hipoteca com o Chase, eles têm um sistema pelo qual você pode pagar por telefone, o sistema reconhece o recebimento de fundos no dia em que o pagamento é processado sujeito a você, fornecendo que você fornece seu número de roteamento e número de conta verificando. Após o recebimento da minha declaração, Chase refletiu que o pagamento foi processado um dia depois e cobrou uma taxa de atraso de {$ 25,00}. Isso aconteceu em XXXX ocasiões para mim este ano. No passado, quando liguei, eles reconheceram uma 'falha nos sistemas', mas desta vez eles não reembolsaram a taxa depois de gastar quase uma hora por telefone e falar com vários agentes. Quantos clientes são taxas cobradas como essa todos os meses, é uma tentativa óbvia de adiar a publicação de pagamentos para obter taxas atrasadas. Gostaria que o XXXX investiga meu caso e a prática geral de negócios aqui. Como o meu é um HELOC, a taxa de atraso é fixada, no entanto, nas hipotecas tradicionais, as taxas tardias podem ser significativamente maiores. Agradeço antecipadamente por sua ajuda a esse respeito.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Isto é para o cartão de crédito xxxx xxxx que termina em xxxx. Eu deveria ter sido autorizado a transferir ou resgatar os pontos XXXX que equivale a pelo menos {$ 590,00} em um crédito de extrato. Em xxxx xxxx, 2017, recebi uma declaração mostrando que os pontos XXXX não estavam disponíveis. Também não posso acessar os pontos online.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Gostaria de apresentar u\n",
            "======================================================\n",
            "Texto: Contestou duas transações e forneceu provas dos reembolsos. Chase está se recusando a creditar meu cartão, cobrou taxas tardias todos os meses - mesmo que o pagamento seja dentro do prazo. Eles precisam creditar os {$ 15000,00} por reembolsos pendentes. E todos os meses de taxas atrasadas em que os pagamentos foram realmente feitos oportunos. Não farei pagamentos em nenhuma conta de cartões com eles até que eles corrigissem.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  \n",
            "\n",
            "Nesse caso, a mel\n",
            "======================================================\n",
            "Texto: Fiquei chocado ao rever meu relatório de crédito e encontrei atraso no pagamento nas datas abaixo: 30 dias atrasado a partir de xxxx xxxx 30 dias atrasado a partir de xxxx xxxx 60 dias atrasado a partir de xxxx xxxx 30 dias atrasado como xxxx xxxx 60 dias depois, como do xxxx xxxx Não tenho certeza de como isso aconteceu, acredito que fiz meus pagamentos quando recebi minhas declarações. Meu único pensamento é que minha declaração mensal não chegou a mim.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Estou profundamente abalado por\n",
            "======================================================\n",
            "Texto: Desde que relatar um problema com minha conta corrente terminando em xxxx para perseguir, sobre qual débito xxxx não autorizado e xxxx inesperado (e novo) {$ 20,00} Taxa de perseguição, enviou minha conta em um estado negativo em cascata, as taxas e as penalidades continuaram a montar. . Em valores que não pago realisticamente, mesmo nele eram justos.   Como Chase pode justificar aproximadamente xxxx dólares em taxas de cheque especial que foram montadas (e foram contribuídas pelas próprias taxas) que todos foram iniciados no débito xxxx não autorizado ... após o qual um punhado de novos débitos ocorreu, não foram pagos e foram então acessados ​​taxas. É injusto e ultrajante, considerando que eu notifiquei a perseguição dos erros e minha intenção de tentar parar o sangramento e Chase simplesmente lançou a desculpa \"Você\" teve reversões no passado, para que não te ajudemos \". Deve ser ilegal o que aconteceu aqui.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  \n",
            "\n",
            "A Chase está ciente\n",
            "======================================================\n",
            "Texto: , Falha no Protetor de Produtos em XXXX 2007, de fato, ilustra como a Chase apenas criou ou comercializou produtos que não eram nada além de fraude do consumidor. O protetor de produtos da Chase também aumentou o risco de falha hipotecária para Chase recusou -se a considerar as demandas XXXX por ajuda de perseguição, enquanto Chase buscava agressivamente coleções em somas protetor de produtos era de fato responsável e, se não responsável, foram comercializados fraudulentamente com ajuda de Chase como produto que não atendia Reivindicações de publicidade ou marketing. Assim, novamente, a Chase e o protetor de produtos estavam enganando xxxx quanto aos benefícios relacionados aos produtos perseguindo avançados à medida que a proteção paga por xxxx. De fato, a falha do protetor do produto no pagamento de reivindicações de interrupção da hospitalização de três meses com base na incapacidade xxxx xxxx devido à hospitalização para devolver as reivindicações enviadas por correio para o endereço XXXX não foi fisicamente devido à hospitalização. O protetor de produtos e o Chase declarados em vigor não foram desculpas para não devolver os formulários de reivindicação por correio para abordar XXXX não ocorreu devido à hospitalização. Representações de produtos de fraude por perseguir e protetor de produtos que perseguiram cheques de perseguição para xxxx xxxx e por meio de anúncio e envie um e -mail para os clientes em todas as linhas do estado. Marketing de produto de fraude em violação da lei da Flórida XXXX, lei de seguros estaduais e outras leis da Flórida em caso de xxxx. Observe a falha da falha do produto Chase em pagar a reivindicação hospitalar de xxxx e a recusa de perseguir em ajudar xxxx xxxx quando o XXXX teve proteção contra fraudes protetoras e proteção de crédito também era negação de benefícios de fraude, XXXX havia pago e disse que ele poderia esperar do Chase. De fato, Chase sabe que não forneceu e declarou o mesmo em correspondência com XXXX, mas não fez nada para corrigir o dano causado a XXXX ou aos investidores. No entanto, Chase usou falsamente relatórios de crédito como desculpa para diminuir a conta do XXXX Home Equity também representações fraudulentas do fato material por motivo real que a conta de patrimônio reduzida é porque a conta foi baseada em representações fraudulentas da fraude de avaliação e, de fato, um empréstimo de mentiroso foi criado pelo Chase. Xxxx xxxx negou ajuda para corrigir a falha do protetor de produtos, do protetor de fraude e do protetor de crédito. Chase disse a XXXX, independentemente de ter sido hospitalizado e formas enviadas por correio para sua residência desocupada xxxx deve pagar que XXXX pagou conforme acordado pelos representantes da Chase Collections. A Chase Products acabou sendo nada além de fraude e ativar a hospitalização do protetor de produtos acabou por não pagar uma reivindicação, conforme prometido antes da hospitalização. Razão dada os formulários não retornam dentro de 30 dias. Bem, a permanência no hospital foi de três meses. A Chase usou esse crédito para relatórios de crédito e para redução na conta do patrimônio líquido, com fundo disponível para uso em xxxx xxxx xxxx a xxxx, sem aviso prévio, causando cheques retornados e grandes danos a xxxx. De fato, a conta do patrimônio foi uma oferta feita a xxxx para não cancelar a nota primária do xxxx 2007. Conta de patrimônio criada apenas pela autoridade verbal. Somente posteriormente em XXXX, o Chase enviou a hipoteca xxxx para assinar duas vezes, e uma carta informando que o perseguidor não precisava de xxxx para assinar. Tudo isso enquanto xxxx hospitalizou fora do estado no Hospital VA. Em breve, a Chase quebrou todos os acordos relativos a mortates atacou agressivamente o XXXX usando muitas cláusulas para que Chase venda a nota primária e feche o risco de patrimônio. A criação da nota primária e da patrimônio líquido em xxxx 2007 foi base em representações fraudulentas e garante, para que o Chase pudesse obter lucros de curto prazo e externalizar o risco de conta de capital do XXXX Dollar aos investidores por meio de venda para xxxx xxxx em xxxx xxxx que permitem que a permitir que ganhe Novamente, atendendo a conta e taxas de cobrança e prêmios de seguro etc. para a conta xxxx. XXXX foi puro e simples de sua casa há mais de trinta anos.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  A conta de patrimônio l\n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, nós (meu marido e eu) tomamos um empréstimo de capital doméstico no valor de {$ 150000.00} também concordamos em comprar um seguro suplementar que cobriria nossos pagamentos nos casos de xxxx, desemprego involuntário, licença de ausência e xxxx acidental. Isso foi considerado o pacote de platina de Chase. Pagamos uma taxa mensal adicional por essa proteção. Em xx/xx/xxxx, perdi meu emprego depois de 22 anos e começamos a percorrer nossa papelada para a casa. Encontramos a papelada que assinamos há 10 anos e descobrimos que nós dois seríamos elegíveis para essa proteção que pagamos por muitos anos. (Meu marido se tornou xxxx xxxx de volta em xx/xx/xxxx depois de cair de uma torre de água no trabalho.) Então fui perseguir para pedir que eles ajudem depois de fazer várias ligações longas sem sucesso. Eles então me informaram que haviam cancelado esse programa em xx/xx/xxxx, perguntei se fomos notificados e eles disseram que haviam enviado uma carta, perguntei se tínhamos reconhecido uma carta como nunca recebíamos uma carta . Eles afirmaram que não sabiam, então pedi uma cópia desta carta que haviam enviado e disseram que não poderiam fornecer essa carta. A única resposta deles é que eu poderia enviar uma carta ao departamento de pesquisa deles. Esta é uma maneira horrível de tratar um cliente que pagou fielmente este contrato por 10 anos !! Eu gostaria de uma determinação para essa situação e sentir que devemos ter alguma resposta! Por favor ajude! Estamos lutando para pagar nossas contas e pensamos que estávamos segurados por esse tipo de situação. Eu posso fornecer qualquer prova necessária, obrigado !!\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Havia duas cobranças feitas na minha conta corrente #: xxxx que contei com o Chase Bank. Uma das cobranças foi cobrada pela XXXX, que é uma empresa de startups que fabrica pequenos motores para a sombra da janela. O site é xxxx, havia um pedido feito para esta empresa no XXXX pelo valor de {$ 1500,00} e a empresa nunca enviou nenhuma mercadoria até agora. A empresa foi contatada sobre o motivo de não enviar o produto, mas eles responderam que o produto não estará disponível e não estava dando nenhum reembolso. Entrei em contato com o Chase Bank sobre obter os fundos de volta desta empresa devido ao fato de que eles receberam o dinheiro, mas não enviariam nenhum produto para mim e eles também não devolveriam nenhum reembolso, o que para mim parece uma empresa fraudulenta. Enviei -lhe o site para o nome da empresa, xxxx, localizado em xxxx, xxxx. No início, o Chase Bank aprovou o valor da cobrança em disputa de {$ 1500,00}, mas mais tarde, uma carta foi enviada para mim afirmando que a cobrança estava correta e o valor de {$ 1500,00} teria sido revertido sem nenhuma explicação lógica. Como uma cobrança pode estar correta quando uma empresa nunca enviou um produto para alguém nos últimos três meses, mesmo quando contatada, a empresa ainda não enviou o produto ao cliente. A segunda carga foi feita de uma loja xxxx xxxx em xxxx, Florida no xxxx em torno de xxxx. O local da loja fica a cerca de três horas de onde eu moro em XXXX XXXX, Flórida, pelo valor de {$ 1500,00}. A cobrança foi feita na minha conta de dentro da loja e o banco me enviou automaticamente uma mensagem de texto do número: xxxx sobre a cobrança {$ 1500.00} e me disseram para responder 1 se sim, e responder 2 se não. Respondi 2 por não, porque não fiz a cobrança e realmente entrei na minha conta através do aplicativo móvel www.chase.com para confirmar a cobrança. Quando respondi à mensagem de texto, o Chase Bank me chamou automaticamente de alguém do atendimento ao cliente falou comigo em torno de xxxx xxxx de xxxx na manhã do xxxx e o representante do atendimento ao cliente me informou que meu cartão de débito era obrigado a ser fechado em ordem para evitar acusações fraudulentas adicionais. No xxxx, entrei em uma loja do Chase Bank e solicitei que um cartão de débito fosse impresso para mim e que o {$ 1500,00} fosse revertido, fui informado pelo representante do Banco do Chase que ela não podia reverter a carga por sua própria E ela ligou para o departamento de fraude do Chase Bank de: xxxx xxxx para configurar uma disputa para a cobrança {$ 1500.00} e também solicitar que um cartão de débito Chase XXXX seja enviado para mim. Algumas semanas depois, o Chase Bank negou a reivindicação, afirmando que eu coloquei a reivindicação no XXXX quando a cobrança, como realmente fez no XXXX. Tentei explicar ao departamento de fraude o motivo pelo qual a reivindicação foi configurada dois dias após a acusação original, mas ninguém estava disposto a me ouvir e alegou que foi minha culpa. Segui todos os procedimentos estabelecidos pelo Chase Bank para iniciar uma disputa, mas o Chase acabou encerrando minha conta bancária e tirou {$ 3100,00} da minha conta bancária, o que impedia que minhas contas fossem pagas. Eu gostaria que o Chase Bank me devolva meu {$ 3100.00} e também restaure minha conta bancária. Não hesite em entrar em contato comigo a qualquer momento por e-mail, xxxxxxxxxxxxx ou por telefone, xxxx. Agradecemos antecipadamente por seu apoio!\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: O Chase Mortgage # XXXX reagiu esse item, violou meus direitos de acordo com a FCRA.   É claro que a primeira data de inadimplência contínua neste item é xx/xx/xxxx. (Consulte a reportagem anexada desta listagem) De acordo com os itens da FCRA, não devem mais relatar após 7 anos.   O Chase MTG agora colocou o item como uma carga com uma nova data recente a partir de xx/xx/xxxx; que claramente é impreciso. Meu relatório anterior de xx/xx/xxxx relatou este item como carregamento em xx/xx/xxxx. Chase reagiu esse item e causou mais danos ao meu crédito.   Anexei cópias de tela de impressão do meu relatório de crédito de xx/xx/xxxx mostrando que eles realizaram o item. Também anexei a tela de impressão do relatório de crédito para relatório de relatório xx/xx/xxxx relatando uma data de carregamento de xx/xx/xxxx. A FCRA afirma claramente que as informações no relatório de crédito devem ser precisas ou as informações imprecisas devem ser removidas. É evidente que não posso confiar em nenhum relatório e solicitar que este item seja removido do meu arquivo imediatamente.   Além disso, o novo relatório deste item como uma nova carga da XX/XX/XXXX parece estar relatórios retaliatórios da Chase, para que este item apareça como uma nova conta de derrogatório para outros credores em potencial (que claramente não é). Uma acusação é uma prática contábil em que os bancos devem mover dívidas colecionáveis ​​envelhecidas da coluna de ativos para fins contábeis; portanto, é um único evento único, não um status que pode ser relatado mês após mês; Como eles não movem a dívida para frente e para trás em seu balanço patrimonial.   A FCRA fornece alívio aos consumidores, pois, após 7 anos, os itens devem ser excluídos do relatório de crédito e também proíbe a má conduta retaliatória dos credores. Em que Chase claramente não tem consideração por seguir os procedimentos, conforme descrito na FCRA.   Atualmente, esse relatório retaliatório está prejudicando e me impedindo de obter meu financiamento pessoal.   Estou contestando isso com a ajuda e apoio do Departamento de Proteção Financeira do Consumidor e solicito que meus direitos sejam protegidos conforme descrito na FCRA. Além disso, essas discrepâncias nos relatórios me levaram a acreditar que seus dados são imprecisos e, portanto, não podem confiar em nenhum de seus dados em relatar as agências de crédito. Como eles podem relatar informações ano após ano e alterá -las. Como posso, como consumidor, confiar que eles relataram informações corretas?   Passei pelo mesmo processo de disputa com xxxx xxxx xxxx que também estava relatando a primeira hipoteca imprecisa. Eles também haviam reagido a listagem, mas agora consertaram os relatórios. Anexado está uma cópia da letra que eles me enviaram datada de xx/xx/xxxx, informando que eles estariam removendo as informações imprecisas.   A única resolução aceitável para este assunto é a exclusão imediata da listagem imprecisa, remova este item do meu relatório de crédito.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:    Estou anexando informaç\n",
            "======================================================\n",
            "Texto: Eu tenho cartões de crédito xxxx com o Chase Bank. XXXX é uma liberdade de perseguição, a outra uma safira perseguida. Eu tive o pagamento automático de ambos os cartões durante a maior parte do tempo em que tive as contas, vários anos neste momento. Recentemente, meu cartão de crédito foi desligado e, quando entrei em minhas contas, mostrou que ambos estavam vencidos. Além disso, o sistema de pagamentos automáticos que eu configurei havia sido desligado, acionando as taxas atrasadas e o cartão fechado. Nunca recebi nenhuma notificação da Chase de que eles haviam desligado meus pagamentos automáticos. Quando liguei para eles, eles me disseram que não tinham certeza, mas que os pagamentos de automóveis podem ter sido desligados porque um dos meus pagamentos saltou há alguns meses (eu estava mesclando contas bancárias com meu marido, o que causou a lacuna). No entanto, desde então, fiz pagamentos usando a mesma conta várias vezes desde então, é uma boa conta e, no momento em que eles desligaram meus cartões de crédito, tinham mais do que suficiente para cobrir meus pagamentos mínimos. Quero ter certeza de que eles não fazem isso com outros clientes - não acho que um banco seja capaz de desligar os pagamentos automáticos e desencadear taxas atrasadas imerecidas sem notificar um cliente. Eles devolveram as taxas atrasadas quando eu reclamei, mas eu poderia ter perdido totalmente se não estivesse passando por minhas acusações.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "Eu aconselho a recl\n",
            "======================================================\n",
            "Texto: O JP Morgan Chase Bank e as empresas de cartão de crédito roubaram meus ativos. Minha avó era xxxx xxxx xxxx e meu pai era xxxx xxxx xxxx e eu sou xxxx xxxx xxxx. Eu procuro uma resolução há 13 anos e não me deram nada além de mentiras e perseguições de ganso selvagem de todas as entidades. A questão parece ser muito impressionante para alguém agir a meu favor. Começarei a postar meus extratos bancários e outras evidências on -line, a fim de conscientizar a comunidade da minha vida desperdiçada e da existência completa, enquanto alguém desfruta dos despojos do sangue, suor e lágrimas da minha família. Minha família foi ignorada contratualmente e deixada para morrer, sou o principal designado e o último da minha linhagem. A conta foi dada a todos os bancos daqui para Nova York. Minhas despesas não me permitem assistência legal neste assunto. Por favor, ajude, fui reduzido por rótulos que não descrevem a verdade. Sofri muito, todos ficamos sem nossos investimentos por mentiras e desculpas do titular por 13 anos. Meu ssn é xxxx\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: -xxx-xxxx e meu númer\n",
            "======================================================\n",
            "Texto: Usando meu xxxx xxxx xxxx, na minha conta xxxx, eu coloco um pedido para {$ 330,00}, número do pedido xxxx. O pedido foi separado em vários pacotes e, por um motivo ou outro, nunca chegou um dos pacotes. O pacote ausente era {$ 37,00}. Quando o pacote foi atualizado para entregue em xx/xx/2020 em xxxx pm sem nenhum caminhão de entrega xxxx que aparecesse o dia todo, eu imediatamente verifiquei minha varanda e outras áreas ao redor da minha casa. No entanto, o pacote não estava lá. Depois caminhei para outro prédio que possui uma sala de correio segura, que às vezes são deixados. No entanto, não havia novamente nada lá. Passei pela minha conta XXXX e usando o portal de atendimento ao cliente, registro de data e hora em xxxx, observei que nenhuma entrega foi feita no meu endereço. Fui informado para esperar dois dias completos e, se o pacote não foi recebido, recebi um reembolso ou o XXXX substituirá meu item.   Depois de esperar os dois dias completos, estendi a mão para xxxx em xxxx/xxxx/xxxx e relatei o item como ainda não foi entregue. Foi -me dito pelo atendimento ao cliente que eles pesquisaram o item e foram instruídos a esperar mais dois dias até xx/xx/2020, e definitivamente receberei o item.   Agora, em xx/xx/2020, entrei em contato com xxxx relatando o item como ainda sendo entregue. No entanto, eles declararam devido a uma entrega observada, que não sou elegível para um reembolso ou substituição.   Apesar da prova documentada do item nunca entregue e a correspondência documentada de xxxx garantindo um reembolso, o XXXX ainda está listando uma cobrança por {$ 37,00}, para um item nunca entregue. Isso ainda aparece no meu relatório de crédito como parte do meu saldo, apesar de não fornecer os serviços para executar a cobrança na minha conta de crédito em primeiro lugar.   Não procurei perseguir, pois uma disputa e/ou estorno não faz sentido. Ao iniciar a transação e XXXX, pode mostrar documentos fictícios de entrega. No entanto, incluí o XXXX na reclamação, pois isso afeta meu XXXX XXXX contratado através do Chase.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:     Espero que o problema seja res\n",
            "======================================================\n",
            "Texto: O cartão de crédito da Chase não paga a APR mais alta quando você efetua um pagamento com cartão de crédito. Então, uma transação {$ 900,00} em dinheiro que recebi há 10 anos acumulou juros para onde todo o meu saldo de cartão de crédito agora é considerado todo dinheiro em uma abril de 26 %!!\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "A Chase oferece diversas op\n",
            "======================================================\n",
            "Texto: Eu pedi repetidamente a identidade, telefone #, endereço do investidor do suposto empréstimo hipotecário (originado por xxxx xxxx), que supostamente ficou securitizado com mais de xxxx outros empréstimos hipotecários em xxxx. Isso é relatado sob um número de manutenção diferente por perseguição de xxxx. Tenho direito sob as novas leis de Dodd-Frank e leis de Tila para saber quem é meu (s) meu (s) credor (s) para que eu possa negociar diretamente com o credor. Chase enviou uma carta datada de xxxx xxxx, 2015, afirmando que o investidor para o qual eles atende é: xxxx como administrador do xxxx e depois listam a pesquisa de hipotecas (xxxx) em xxxx xxxx, mn xxxx e o telefone xxxx.   Este não é o investidor e o Chase continua a confundir e ofuscar a verdade. Não se deve ser o administrador da confiança securitizada .... não é o investidor. Além disso, quando eu chamo esse número -eles não falam comigo e me dizem para falar com o Chase. Eles disseram que são indenizados .... significando protegidos pelo seguro de Chase. Eu preciso falar com o investidor. Eu preciso ser capaz de negociar diretamente com o investidor. Chase não está cumprindo com Dodd-Frank, nem Tila. Eles estão me deixando de pedra. Não é mais \"proprietário\" quem é o investidor. Não vou tolerar a parede de pedra de Chase. Chase está agora quebrando as leis mais uma vez.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "\n",
            "Eu irei entrar em cont\n",
            "======================================================\n",
            "Texto: Data: XX/XX/2019 Transação: XXXX Valor: {$ 80,00} Recebi um produto que não corresponde à descrição. (Eles alegaram que eram luvas de lagosta, mas não são.) O vendedor não responde, então tentei ir através da minha empresa de cartão de crédito, mas não há como entrar em contato com a empresa do cartão de crédito. Tentei registrar uma disputa on -line e no aplicativo, mas ela não permite. Gostaria que o cartão de crédito resolva isso com o comerciante e credite meu cartão pelo valor total.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Recebi uma carta datada de XX/XX/2020 Oferta para refinanciar minha hipoteca com o Chase Bank a uma nova taxa de juros de 3,625 %. A primeira página declara com destaque \"Não há custo para você\", que é repetido três vezes na primeira página. A segunda página declara \"{$ 0,00} custos de fechamento '' e uma lista detalhada de custos de fechamento de amostra com custos XXXX. Respondi à oferta aplicando e recebi um telefonema de um consultor de mortalidade do Chase. Ele me fez algumas perguntas e depois me deu alguns termos em uma manutenção refinanciada com um pagamento mensal mais alto do que o que estou pagando atualmente. Após o interrogatório repetido, ele me disse que a mortama incluiria cerca de {$ 2700,00} nos custos de fechamento. Eu contei a ele sobre a carta, duas vezes. Ele finalmente apenas admitiu que eles significavam \"nenhum custo de fechamento na frente\". A carta não diz \"custos de fechamento dobrados no diretor\", diz \"sem custos de fechamento\". Eu disse a ele que estava retirando minha inscrição e pendurou acima.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "\n",
            "Meu conselho é se inform\n",
            "======================================================\n",
            "Texto: Peguei um empréstimo {$ 15000.00} isento de juros (até que xxxx de xxxx) da minha conta de ardósia do Chase para melhorias da casa, que foi refletida na minha instrução xxxx xxxx, xxxx. Havia uma taxa de transferência de saldo de {$ 300,00}, que eu paguei, juntamente com minhas compras de {$ 90,00} e também um pagamento de {$ 1100,00} para o empréstimo {$ 15000.00}. Então, na minha instrução xxxx xxxx, xxxx, fui cobrado {$ .00} juros sobre compras. Quando liguei para perguntar, me disseram que não importava o quanto eu paguei todos os meses, mesmo que fosse muito no mínimo, eu seria cobrado juros (11,49 %) nas minhas compras. Disseram -me que essa era uma \"conta giratória\" e que não há nada que eu possa fazer para não ser cobrado juros pelas compras, a menos que eu pague o saldo inteiro. Acabei de pagar novamente e estou indo Para ligar para ver se eles me permitirão aplicar uma quantidade de minha escolha em relação às minhas compras durante o último mês. De qualquer forma, não vou usar esse cartão de crédito para compras novamente. Cuidado com essas ofertas de \"juros de zero por cento\" '! Eles são enganosos.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Meu marido e eu temos uma conta de cheque conjunta com Chase. Meu marido tem essa conta há 7 anos. Chase encerrou nossa conta com nossa notificação anterior hoje xx/xx/2018. Quando liguei para descobrir o motivo da única resposta que eles me dariam é \"Chase revisou sua conta e decidiu fechá -la '', meu marido e eu estávamos esperando um depósito direto de nosso emprego hoje e quando perguntei sobre isso O dinheiro Chase respondeu que está segurando o dinheiro por dois dias úteis e depois enviará um cheque. O dinheiro na minha conta corrente é o único dinheiro que tenho. Eu tenho uma filha de XXXX Ano em casa que preciso alimentar e pagar Sua creche. Não posso esperar para que um cheque venha pelo correio. É insondável para mim que Chase tenha permissão para segurar meu dinheiro e nem me dar uma explicação. Chase até abriu uma conta poupança para mim em xx/xx /2018, então não houve problema com minha conta ou minha conta. Entrei em uma filial para ver se eles podem me fornecer mais informações e eles disseram que \"às vezes o Chase faz isso e eles nem mesmo fornecem informações aos gerentes da filial. '' O gerente da filial também não viu nada sobre nossa conta.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "\n",
            "Estou extremamente decepcion\n",
            "======================================================\n",
            "Texto: Esta queixa é contra o cartão de crédito Chase Freedom Visa. Eu o uso há mais de um ano, mas comecei a ter problemas com este cartão. Eu vejo algumas das transações que não me pertencem nem foram autorizadas por mim. Eu realmente aprecio se você pudesse resolver esse problema. Fiz uma queixa para perseguir a liberdade diretamente e eles cancelaram o cartão. Um cartão de substituição foi emitido para mim, no entanto, ainda vejo as cobranças na minha conta e não foi demitido. Eu gostaria de solicitar u para cancelar as seguintes cobranças não autorizadas: 1. em xx/xx/xxxx - {$ 520,00} xxxx xxxx xxxxxxxx 2. em xx/xx/xxxx - {$ 370.00} xxxx xxxx 3. Em xx/xx/xxxx - {$ 21.00} xxxx 4. Há uma carga de {$ 59,00} em nome de xxxx em xx/xx/xxxx, 2 cargas de {$ 140,00} em xx/xx/xxx e poucas cargas de { $ 16,00} no mesmo dia. Eu realmente não os entendo. Por favor ajude!\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "Eu compreendo sua frust\n",
            "======================================================\n",
            "Texto: Chase foi notificado de várias compras que a Amazon alegou que foram feitas por conta da minha conta. A resposta ao Chase foi difícil, pois a Amazon removeu o acesso à conta, pois a empresa tomou uma decisão arbitrária de que eu tomei um retorno de um produto que deveria ser plug and play. A Amazon está tendo problemas na minha área onde as entregas não estão acontecendo e ou acabam sendo levadas por pessoas que passam. A empresa assumiu uma posição de que fará qualquer entrega que eles alegam que seja feita para estar na posse da pessoa que ordenou o item. Tentei resolver isso com a Amazon apenas para saber que a empresa não me ajudará, pois minha conta foi fechada e quaisquer compras que fiz não foram visíveis para recebimentos ou garantias. Chase recebeu as informações sobre as entregas e os itens que não chegaram. Coloquei a reclamação como o fornecedor que teve o problema é a Amazon. O item enviado via XXXX ou XXXX não teve problemas. Eu forneci as informações que suportam a situação em que os pacotes foram roubados ou nunca entregues ao Chase. O processo de obtenção das informações foi difícil, pois o Chase não teve acesso aos recibos da Amazon. As informações fornecidas eram uma foto de um pacote em uma varanda. Minha varanda parece a mesma que a varanda dos vizinhos. Solicitei que a Amazon deixe itens na caixa de bloqueio ou deixe uma nota e nenhum deles ocorreu. As respostas que recebi foram que os itens quando restam agora são considerados entregues. Isso não faz sentido, pois não protege os itens de serem roubados, se forem entregues. As transações que procurei que eu possa ver não estão em posse e a Amazon não discutirá. Chase optou por escolher a Amazon devido ao seu relacionamento com eles para ser o provedor de cartão de crédito da Amazon que ganha muito dinheiro com a empresa.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Último xxxx, 2014, eu estava verificando meu saldo da conta da Chase Checking online. Para minha surpresa, notei uma acusação de aproximadamente {$ 1500,00}, que estava tentando ser colocado. Felizmente, não passou. Não consegui/não consegui identificar quem estava tentando colocar a acusação. Liguei para Chase, com quem eu banho. Disseram -me que eles não podiam identificar quem estava tentando tentar tirar dinheiro da minha conta corrente, sem meu conhecimento. Eu garanti a eles que não autorizei a acusação e eles me garantiram que a acusação não passaria e que eles reverteriam a acusação de serviço que eu foi acusado. Ok, problema resolvido. Longe disso. Eu tenho uma hipoteca, com meu pai, em uma propriedade de aluguel que ele possui. Meu pai está envelhecendo e seu xxxx, então eu tenho cuidado de seus negócios pessoais/comerciais. Para minha surpresa, recebi a declaração de hipoteca da empresa de hipotecas que detém a hipoteca/nota sobre a propriedade de aluguel. Eles foram os que tentaram realizar um pagamento em minha conta para um pagamento de hipoteca; Perseguir hipoteca. Entrei em contato com eles para ver por que eles tentaram fazer isso sem meu conhecimento. Eles me disseram que entraram em contato com meu pai, o que meu pai não se lembra, e que ele havia autorizado o pagamento, fora da minha conta, que meu pai não tem o nome dele, nem tem autoridade para retirar nenhum fundos a partir de. Chase Banking não conseguiu identificar a hipoteca Chase ????? Entrei em contato com a Chase Mortgage para tentar resolver esse assunto amigavelmente. Conversei com vários representantes da Chase Mortgage e solicitei que uma resolução fosse perdoar os aproximadamente {$ 1500,00} que estávamos atrasados ​​em nossos pagamentos. Eles afirmaram que não tinham autoridade para fazer isso e entrar em contato com a Chase Corporate. Eles me deram as informações de contato adequadas, então pensei. Xxxx, sem resposta; Xxxx, sem resposta; Xxxx, sem resposta da Chase. Recebi uma carta de uma empresa de cobrança em xxxx; Eles disseram que são um escritório de um advogado. Se este é um escritório de um advogado, isso explica muito sobre o nosso sistema jurídico. Nos dois meses seguintes, eles enviaram cartas de inúmeras pessoas, a quem tentei entrar em contato, sem resposta deles. Finalmente, recebi uma ligação e pude conversar com alguém que me garantiu, que eles eram a pessoa de contato. Eu dou crédito a ele, Chase Corporate, voltou para mim. A descoberta deles era que eles não fizeram nada de incorretamente. Sério ; A pessoa com quem falei em XXXX, após uma investigação, afirmou que agiu de forma inadequada. A propriedade está atualmente em execução duma hipoteca, que é uma questão paralela. Se a Chase Corporate tivesse tentado resolver esse assunto, nós (nós ou Chase) não estaríamos nessa situação. Isso explica muito sobre por que nosso sistema bancário está na forma em que eles estão. Eu tenho todos os contatos, documentação, os contatos com os quais conversei e as datas. (Chase, empresa de coleção e suposta empresa de advogados.) Quantas outras pessoas isso aconteceu? Quantas outras pessoas têm os meios para buscar isso? Pior então, quantas outras pessoas não sabem que foram enganadas?\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta:  Esta experiência foi m\n",
            "======================================================\n",
            "Texto: Em xxxx, meu marido morreu e todos os credores foram notificados. Chase não me permitiu acesso à conta HELOC que eles aparentemente haviam comprado em um pacote, nem falariam comigo. Eu nunca estive em um banco de perseguição. No entanto, continuei pagando todas as contas como meu marido havia feito. Nos homens xxxx ou xxxx xxxx bateu na minha porta e me disse que eu precisava desocupar minha casa porque estava em execução duma hipoteca. Os homens estavam vestidos como deputados do xerife em camisas bronzeadas e calças marrons. Minha única opção era assinar os envelopes xxxx xxxx que eles me entregaram. Não havia caminhão xxxx. Não havia veículo da XXXX. Quando protestei que pode ser impróprio assinar para um homem morto, eles insistiram que eu deveria. Captei, assinei um homem morto e recebi os envelopes. Abrindo o envelope que me foi abordado, suportei complacentemente xxxx ou sete meses de telefonemas semanais de adolescentes em um call center em algum lugar deste país onde o nome \"Chase\" é pronunciado \"xxxx\". Eles me disseram que eu deveria primeiro refinanciar minha primeira hipoteca que foi comprada por xxxx xxxx. XXXX XXXX ficou feliz em me refinar e fez tudo o que era \"exigido\" pelos regulamentos xxxx. Claro, eles ficaram felizes em ajudar. Abaixou meus pagamentos mensais com um novo empréstimo de 40 anos, estendeu a data de maturidade a xxxx, quando eu estarei comemorando meu aniversário xxxx em uma casa em que vou fazer pagamentos há 64 anos. A certidão de óbito do marido morto enviou por fax antes de receber a chamada giggly e feliz do meu \"especialista em preservação doméstico\", oferecendo -me opções xxxx. Xxxx. Pague -lhes {$ 50000.00} imediatamente ou XXXX desocupado. Veja bem, eu estava fazendo pagamentos para perseguir continuamente, eles me bloquearam do site deles e me recusaram a me enviar uma declaração. Eu tinha uma úlcera sangrando a essa altura e disse ao meu especialista em preservação de casa adolescente para não me chamar mais. Ela parecia genuinamente perplexa. Eu continuava pagando. Em xxxx deste ano, recebi novamente envelopes xxxx xxxx, mas desta vez por um funcionário XXXX real 'que bateu em voz alta deixou os envelopes XXXX na porta, como os reais. Dentro do meu envelope, eles decidiram agora me oferecer um empréstimo de 40 anos com a atração de uma taxa financeira de 1 % - - o primeiro ano. Em nome de todas as coisas decentes e honestas, devo perguntar, como isso ainda pode estar acontecendo? Eles usam palavras como \"O tempo é essencial '' e continuam a acompanhar ligações de crianças que não têm idéia de que tipo de trabalho predatório estão fazendo.\" Divisão de Escalação ''? Contabilidade online ilegível? Este tem sido um pesadelo para a mulher honesta que eu sou. É um horror. Eu enviaria uma cópia disso para xxxx xxxx, mas não é engraçado. Eles sempre se recusam a reconhecer a morte do meu marido, nem relatarão os juros pagos ao meu número de previdência social. Eles continuam a usar o dele, por isso, para sempre, receberei avisos anuais do IRS de que reivindiquei juros não me devidos. Graças a xxxx, o IRS tem pessoas brilhantes e honestas em seu emprego, que prontamente veem onde surgiu o problema.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Eles me corrigiram no mesmo\n",
            "======================================================\n",
            "Texto: Em xx/xx/2019, verifiquei minha conta de corrente militar do Chase Bank e observei a {$ 340,00} deduzi da minha conta corrente na qual havia apenas {$ 83,00} na minha conta. Entrei em contato com a advogada tributária do IRS MS.XXXX na minha modificação de pagamento de {$ 180,00} por mês em que está sendo revisada pela administração do IRS. Os pagamentos de {$ 340,00} não deveriam ter sido deduzidos em xx/xx/2019 e as taxas de cheque especial não deveriam ter sido implementadas. O advogado tributário do IRS está trabalhando em meu nome nos acordos de pagamento do IRS de {$ 180,00}. Ela pode ser alcançada art xxxx.   Depois que o IRS concordar na modificação do pagamento de {$ 180,00}, os fundos estarão disponíveis na minha conta corrente. Eu sou xxxx xxxx e aposentado. Estou com uma renda fixa. Assuntos urgentes muito extremos!\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Meu nome é e acho que as práticas de cobrança de cartões de recompensas da Amazon.com são enganosas e confusas. Entrei em contato com a empresa no meio do xxxx para notificar que perdi minha carteira e, como cortesia, eles cancelam a taxa tardia e eu faria os dois pagamentos em xx/xx/xxxx. Foi -me dito pelo representante de atendimento ao cliente em xxxx que essas informações seriam incluídas na minha conta e eu não seria cobrado a taxa de atraso. Recentemente, fiz um pagamento {$ 60,00} em xxxx e recebi e sobrecarreguei {$ 25,00} por não incluir {$ 4,00} no pagamento {$ 60,00}. Agora estou pagando uma taxa de atraso {$ 35,00} por {$ 4.00}. Isso é usura e não posso pagar esse pagamento. Liguei para o Amazon Chase XXXX/XXXX/2016 e expliquei isso a XXXX, que só me ignorou e continuou insistindo que pago a taxa de atraso {$ 35,00}. Fui cancelado da taxa de atraso xxxx apenas para ser cobrada com outro xxxx. Eu acho que isso não é justo e de práticas comerciais sólidas da ONU. Por favor me ajude porque a Amazon Chase está tentando me arrancar. O número da minha conta termina em xxxx. Muito obrigado pela sua ajuda! Xxxx xxxx\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Sob penalidades de perjúrio, está e a fabricação de declarações que o JPMorgan afirma que eles têm uma nota promissória original e hipoteca assinada por mim para eles, que é uma fabricação e falsa mentira. Eu nunca assinei nenhum documento com eles. Por mais de 12 anos, eles me ligaram e me perseguem 2 e 3 vezes ao dia, mais de 3000 ligações telefônicas. Veja: xxxx, xxxx. Sofremos de xxxx xxxx agora, xxxx xxxx xxxx, nossos netos estão distintos agora, isso está em toda a Internet como delinqüentes. Tudo o que fiz foi obter uma divulgação de crédito de visto de Washington Mutual, que foi encerrada em XX/XX/2008 e nunca designada para ninguém. Por favor, ajude a ajuda. Estamos aposentados e não podemos mais lidar com isso, por favor me ajude ... Eles chamam meus filhos de meus netos, meus vizinhos, eles batendo na casa de todos que nos assediam todos os dias, ajude -nos a detê -los, somos velhos e aposentados e não podemos lidar com isso ...\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta:  eu nunca assinei nenhum\n",
            "======================================================\n",
            "Texto: Recebi uma cobrança por {$ 300,00} em xxxx de xxxx xxxx, para o qual era uma carga não autorizada. Liguei para o Chase Bank e os informei dessa acusação. Eles me emitiram um novo cartão bancário e tentaram recuperar os fundos, mas não tiveram êxito, e é por isso que estou arquivando essa disputa para tentar recuperar acusações.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "Comecei a disputar im\n",
            "======================================================\n",
            "Texto: Todo mês, meu equilíbrio seria o Exemplo A, mas meu beleza fanático na minha conta está no xx/xx/xxxx.chase Bank afirmaria que meu equilíbrio príncipe é um, mas quando eu recebo minha declaração mensal. Vejo que o Chase Adicione Faniacial Interior ao meu Equilíbrio, fazendo com que seja mais alto, então, quando eles fazem a perseguição matemática, sai com mais. Para um exemplo, veja abaixo xxxx em xx/xx/xxxx deve ser xxxx mais introdução. do que o Interess adicionou novamente dizendo que Bal era xxxxon xx/xx/xxxx. Chase em uma posição melhor contra os clientes. Eu acho que é fraude. E mantém o cusumer em dívidas. Você pode exigir seu equilíbrio, mas o dia de interesse adiciona que o equilíbrio é maior por algum motivo. Se eu não disser nada, os bancos estão se safar. O BAL principal altera que muito FRON XX/XX/XXXX para XX/XX/XXXX e Cartão não estão sendo usados.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Quando eles fazem a perse\n",
            "======================================================\n",
            "Texto: Número da reivindicação: XXXX Data do arquivo: XXXX Tipo de reclamação: Trip Cancelamento/interrupção Solicitação de reclamação: PEND Detalhes da reivindicação: Informações do incidente: Data do cancelamento/interrupção: xx/xx/2019 país: EUA Estado: NY Resumo Descrição: Minha filha xxxx xxxx foi diagnosticado com xxxx xxxx xxxx e não foi capaz de fazer a viagem. As seguintes cobranças devem ser reembolsadas por ela: passagem aérea não reembolsável de xxxx: {$ 510.00} Hospedagem: {$ 48,00} Tours: {$ 20,00} Total: {$ 580,00} Detalhes da viagem: Data de partida: xx/xx/2019 Data de retorno: xx//xx/ Reclamação xx/2019 apresentada contra a transportadora: y reembolso da transportadora: nome da transportadora: xxxx xxxx Quantidade de transportadora: {$ 510.00} USD\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  data do reembolso: N/A\n",
            "======================================================\n",
            "Texto: Recebeu cartas de marketing uma vez por mês da Chase nos últimos 18 meses para solicitar o cartão de crédito Chase XXXX. Atualmente, quatro dessas cartas estão em mãos. Todos anunciam a capacidade de solicitar um cartão de crédito com um determinado bônus de inscrição e declarar que você será considerado se não tiver o cartão atual, nem tiver recebido um novo bônus de membro do cartão nos últimos 12 meses. Nenhuma outras limitações são declaradas nas correspondências que limitam o pedido ou garantindo a rejeição para certa subseção de candidatos.   A carta de rejeição do aplicativo foi enviada em xx/xx/xxxx e recebeu no xxxx. Nenhuma reconsideração estava implícita e nenhuma informação de número ou contato foi listada. Depois de reler a parte de trás da carta e pesquisar on -line, descobri que eles têm uma política de negar e não reconsiderar todas as pessoas com 5 ou mais novas contas nos últimos 24 meses. Essas informações não foram listadas em nenhum lugar do site ou no aplicativo. Eles afirmam que não é uma política, mas uma razão de rejeição. Eles não ofereceram nenhuma prova de que não era uma política ou regra difícil. Isso impactou adversamente minha pontuação de crédito com base em uma política não escrita que afirmava ser um segredo comercial quando atualmente é de conhecimento público. Isso também constitui publicidade falsa, pois implica em consideração quando nenhum se destina. Eles também estão coletando informações financeiras em detrimento do candidato sem a intenção de oferecer o cartão, mas a um subconjunto de indivíduos por motivos não créditos. A razão de rejeição original também discrimina a idade, uma vez que as pessoas mais jovens abririam mais contas dentro de um período de 24 meses do que indivíduos mais velhos e estabelecidos.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Para denunciar essa public\n",
            "======================================================\n",
            "Texto: Eu me inscrevi na conta do Marriott XXX sob a promoção \"Você está convidado a aplicar e ganhar 30.000 pontos de bônus depois de gastar {$ 1000,00} em compras nos seus primeiros 3 meses após a abertura da sua conta. '' Não apenas gastei os necessários {$ 1000,00 } Atenda às condições no período de três meses declarado. Apliquei a placa xxxx em xx/xx/xxxx; recebi o cartão em xx/xx/xxxx e fiz as compras entre xxxx e xx/xx/xxxx. Esperei até xx/xx/xxxx antes de fazer a primeira chamada solicitando meus pontos de bônus ao atender ao requisito. Fui informado em xxxx por meio de uma carta que a promoção listada na minha conta foi de 50.000 pontos se eu gastar xxxx, no entanto, liguei de volta com prova de que a oferta de bônus de 50.000 não começou até xx/xx/xxxx, pois recebi tudo de tudo seus e -mails promocionais. Eu pensei em afirmar que tinha provas verificáveis ​​da promoção em que entrei quando tirei uma captura de tela quando me inscrevi. Esperei até XXXX pela mesma negação da empresa. Liguei para um tempo XXXX pedindo para enviar minha prova. Em xxxx, enviei a prova da captura de tela que tirei da promoção e do representante com o qual conversei até confirmado (as chamadas são gravadas) que a promoção que afirmei era na verdade a promoção no lugar XXXX. Mesmo com a prova de que a empresa ainda nega a promoção e a honra da promoção que foi anunciada em xxxx; A razão pela qual eu me apliquei. Desde então, descontinuei meu uso do meu cartão Marriott Bonvoy enquanto me recuso a usar um cartão cuja empresa se recusa a honrar suas promoções anunciadas. Eu atendi o requisito e acredito que eles mudaram as condições da promoção sem notificação para evitar a concessão dos referidos pontos de bônus. Quero alertar outros consumidores para manter registros, tirar capturas de tela (mesmo que tentem negar a promoção) e manter essas empresas de cartão de crédito em seus anúncios.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: O representante da conta bancária do Chase Bank xxxx xxxx hoje à noite me aconselhou que o sistema deles está inativo e eles não estão notificando ou listando débitos feitos na minha conta corrente. Isso é um problema, porque as pessoas podem estar roubando dinheiro de mim sem meu conhecimento. Ele também disse que eles nunca me notificam sobre essas transações. Eu os peguei hoje à noite, mas isso poderia estar acontecendo regularmente.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "\n",
            "Isso é muito preoc\n",
            "======================================================\n",
            "Texto: Comprei um XXXX de um vendedor pela Internet em xx/xx/18. Enviei duas transações para {$ 300,00} e {$ 250,00} para um total de {$ 550,00} (que foi o custo do telefone) através do aplicativo bancário xxxx. Recebi uma fatura de confirmação, mas o vendedor bloqueou meu número e o Instagram e não consigo contatá -los. Tentei solicitar {$ 550,00} de volta deles através do aplicativo XXXX, mas fui informado pela Chase que eles recusaram o pagamento. Fiquei convencido de que a transação seria apoiada pelo meu banco, pois estava vinculada à minha conta Chase. O vendedor não respondeu aos meus textos ou chamadas e não emitiu um reembolso pelo valor. Não recebi o produto e sei que isso foi uma farsa. Entrei em contato com Chase e eles me informaram que não podiam fazer nada porque iniciei uma transação. Eles não cancelaram o pagamento ou interromperam nele. Tentei entrar em contato com o XXXX, mas não consegui chegar ao departamento de atendimento ao cliente. Eu acho injusto que o XXXX não seja apoiado pelo Chase Bank, mesmo que possamos enviar transações diretamente do aplicativo.   Anexei as confirmações que recebi do vendedor e da fatura. Também anexei o e -mail que me foi enviado por Chase.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Por favor, ajude-me a res\n",
            "======================================================\n",
            "Texto: Como as seguintes trocas de email de \"Wed, xx/xx/xxxx em xxxx xxxx\", com Chase demonstrar que tentei o meu melhor para resolver o problema de perder alguns meses de pagamentos há cerca de sete ou oito anos atrás. Mais uma vez, no final de xx/xx/xxxx, tentei resolver esse problema com xxxx xxxx xxxx xxxx xxxx, onde meu número de conta é: xxxx.   Nos dois casos depois que enviei essas instituições exigidas documentos, recebi carta de cortador de cookies. No XX/XX/XXXX Quando tentei resolver esse problema de pagamentos tardios ou ausentes com o Chase, troco muitos e -mails com a equipe do Chase. Por suas demandas de documentos que eu lhes enviei mais de uma vez por suas declarações enganosas, o funcionário da Chase parece ser treinado deixando as pessoas cansadas da corrida. Eu simplesmente não queria mais ir a essa experiência desagradável com nenhuma instituição financeira. Enquanto isso, eu continuava enviando cerca de US $ 100 para o pagamento extra com minha hipoteca mensal para compensar o par de meus pagamentos ausentes sete ou oito anos atrás. Apesar disso, o Chase e agora o XXXX continuam me alegando para o pagamento atrasado que, de acordo com o cálculo, equivale a milhares de dólares.   Recentemente, decidi mudar minha empresa de hipotecas. Na sexta -feira passada, fui para xxxx xxxx xxxx com minha declaração de hipoteca. Disseram -me que o Chase e o XXXX estão relatando às empresas de crédito que estou atrasado para pagar minha hipoteca mensal há vários anos. E até que o XXXX limpe esse defeito do meu registro, anotando milhares de juros e cobranças tardias e acerte meu recorde com as empresas de crédito, o xxxx xxxx xxxx não pode me ajudar. Espero que você possa me ajudar antes de registrar uma suíte de direito contra o XXXX no Tribunal Superior XXXX. Muito obrigado.   ---------- Mensagem enviada ---------- De: xxxx xxxx Data: Sun, xx/xx/xxxx em xxxx xxxx Assunto: Re: Recebi uma carta da perseguição escrita em xx/xx/xxxx declarando documentos necessários re: Por favor, seja meu consultor no pedido que eu já havia apresentado com a perseguição em relação à minha conta # xxxx Para: xxxx   Caro Sr. XXXX.   Eu queria saber se você ainda está com a perseguição? Xxxx xxxx xxxx   Re: Recebi uma carta da perseguição escrita em xx/xx/xxxx declarando documentos necessários Re: Por favor, seja meu consultor no pedido que eu já havia apresentado com a perseguição em relação à minha conta # xxxx   Na quarta -feira, xx/xx/xxxx em xxxx xxxx, xxxx xxxx escreveu: Querido xxxx xxxx,   Na quarta -feira, xx/xx/xxxx, você me disse que eu enviei todos os documentos necessários, mas recebi uma carta da Chase escrita em XX/XX/XXXX, informando os documentos necessários: Propriedade Concluído xxxx, depoimento de dificuldades de adequadamente concluído, documentação de renda. Por favor, envie -me uma linha informando o que você me disse ao telefone que a perseguição tem todo o necessário   Documentos meus e não preciso enviar nenhuma outra documentação. Muito obrigado.   Muito sinceramente,     Xxxx xxxx xxxx     Na sexta, xx/xx/xxxx em xxxx xxxx, xxxx xxxx escreveu: Xxxx xxxx xxxx xxxx Conselheiro, Chase Homeownership Center Xxxx xxxx xxxx xxxx, xxxx xxxx xxxx, xxxx xxxx   Ph xxxx Fax xxxx   Caro Sr. XXXX,   Na quarta -feira, xx/xx/xxxx, conheci você em seu escritório e depois enviei por fax a cópia da minha declaração de imposto sobre minha conta ou empréstimo # xxxx. Como eu disse a você em nossa reunião, o número Chase XXXX poderia me levar a um espera por muito tempo e, eventualmente, cortar, se eu não desligar. Peço que você obtenha minha pasta e trabalhe nela. Pelo menos, posso entrar em contato com você por telefone e/ou vejo você também, se for necessário. Informe -me se você poderia ser meu consultor sobre o pedido que eu já havia apresentado com a perseguição. ... Por favor, envie -me uma linha sobre o meu pedido, se puder.   Muito obrigado pela sua ajuda.   Muito sinceramente, Xxxx xxxx xxxx xxxx, xxxx xxxx xxxx xxxx, xxxx xxxx   C xxxx H xxxx\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta:  F xxxx\n",
            "======================================================\n",
            "Texto: A apresentação de uma nova reclamação porque Chase encerrou minha reclamação anterior sem uma resposta (fiz 6 telefonemas para seus escritórios executivos que não foram devolvidos: xx/xx/xxxx, xx/xx/xxxx, xxxx, xxxx, xxxx e xx/xx /Xxxx de 2019). Em xx/xx/xxxx2019, recebi um documento tributário enviado para minha casa, endereçado ao nome da minha mãe no meu endereço residencial. Minha mãe e eu temos uma conta corrente conjunta juntos, mas ela nunca viveu no meu endereço e sempre recebe declarações enviadas por seu próprio endereço. Eu entrei na minha conta chase.com para ver se Chase havia de alguma forma misturar nossos endereços. Fui à conta conjunta que ela e eu compartilhamos e cliquei em declarações. Quando a declaração do PDF foi aberta, ela continha o nome da minha mãe, seu endereço residencial e detalhes completos sobre todas as seis contas de perseguição, não apenas as que ela e eu compartilhamos. Liguei para Chase e relatei esses dois problemas: (1) o correio enviado para o endereço errado; e (2) a violação da privacidade no site. Os agentes de atendimento ao cliente não pareciam pensar que nenhum desses problemas era preocupante. Eles nos disseram que poderíamos corrigir a divulgação de dados nos emitir, fazendo com que minha mãe desenhasse suas contas para que nem todas aparecessem na declaração combinada. Então, essencialmente, Chase está nos dizendo que eles divulgarão dados financeiros (por meio de uma declaração combinada de PDF) a alguém que não é um titular de conta (eu), a menos que tomemos medidas especificamente para optar por não participar. Eles me disseram que há uma divulgação dada a alguém que escolhe uma declaração combinada. A redação da divulgação não deixa claro que os dados para todas as contas serão compartilhados com um indivíduo que é um titular de conta conjunta em apenas uma conta. Eu arquivei uma reclamação e liguei de volta várias vezes para verificar o status, por um total de 3 horas e 19 minutos do meu tempo, sem resolução da Chase, então agora estou me voltando para o CFPB. Isso é descuidado de desrespeito a dados financeiros pessoais. Estou extremamente decepcionado com a Chase por não proteger as informações financeiras privadas de minha mãe e por não tratar esse assunto com nenhuma urgência.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Recebo aviso do Chase sobre a execução duma hipoteca tentou fazer modificação de empréstimo. A data das salas foi xxxx xxxx, 2017, a modificação foi negado. Antes disso, eu estava trabalhando com uma pessoa que fez um processo para evitar a execução duma hipoteca, quando fui ao advogado de falência para arquivar o capítulo XXXX, ele foi informado de que havia um interrama sobre propriedade e a falência não foi concedida mangueira XXXX, 2017. O problema não era com o advogado de falências, mas com o primeiro cavalheiro. Porque minha propriedade foi anexada a outra pessoa que Chase vendeu\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta:  o empréstimo. Eles n\n",
            "======================================================\n",
            "Texto: Número da caixa FHA xxxx xxxx venda em casa fechada xx/xx/xxxx. Juros cobrados pelo JP Morgan Chase em xx/xx/xxxx. Reembolso esperado por interesse não merecido. O Bank disse nenhum reembolso devido aos regulamentos da FHA.   Empréstimo aberto desde xx/xx/xxxx.   Procurando reembolso de juros um pouco menos que {$ 600,00}.   Espero anexar cópias do extrato de pagamento bancário, refletindo os juros cobrados e a declaração de liquidação confirmando a data de fechamento.   Xxxx xxxx xxxx\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "Infelizmente, não\n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, recebi um aviso de transação de fraude da Chase de que alguém usou meu cartão de crédito da liberdade em Nova Jersey enquanto estou em xxxx. Depois que verifiquei a massagem, liguei de volta e relato essas são as cobranças de fraude e declínio. Muitas transações (e todas as transações são apenas duas quantidades.) Foram feitas e eu nem entendo como, porque eu literalmente consegui o meu próprio no bolso. Após o relatório de fraude, recebi um novo cartão de crédito. No entanto, em xx/xx/xxxx, o Chase me rejeitou algumas das transações porque era válido. Então enviei minha conta de hotel, conta de voo, outras contas de cartão de crédito naquele dia e identificação com minha assinatura e foto para todos os departamentos relacionados à segurança. Por fim, recebi telefonema de uma pessoa do Departamento Executivo (xx/xx/xxxx) que tenho que pagar essas transações de fraude porque 1. é válido e pago com o Chip. 2. Seu cartão não está perdido.   Eles nem podiam explicar bem sobre o porquê, mas continuaram dizendo que é válido e desculpe por essa decisão.   Para mim, é difícil entender por que Chase não me protege. Aqui estão as razões pelas quais eu realmente chateado e você deve cuidar do meu caso. 1. Relatei todas as transações de xxxx foram fraudes logo após verifiquei a mensagem de alerta. 2. Eu e toda a minha família estávamos em xxxx e todos os documentos provavam isso. 3. Não preciso saber como o ladrão copiou e usou meu cartão de crédito, mas Chase tentou me fazer entender/aceitar que é válido porque ninguém poderia copiar chips. A explicação deles me sentiu extremamente decepcionada porque parece que eu menti. 4. O especialista me disse que xxxx verificou que alguém usou o cartão. No entanto, isso não significa que eu usei meu cartão de crédito. 5. Não acho que o departamento de segurança do Chase realmente tenha investigado meu caso. A quantidade total de dinheiro foi superior a {$ 2000,00} ({$ 2300.00}) e cada um acabou {$ 400,00}. O que eu entendo é a assinatura necessária ao visto ao fazer pagamento em xx/xx/xxxx. (Mas Chase me disse que depende do comerciante.) XXXX também deve exigir assinatura. Portanto, o Chase pode corresponder à assinatura no recibo que o XXXX forneceu. Além disso, eles têm câmera de segurança. Por que Chase ignorar todas essas evidências claras de investigação? Sinto que o Chase hesite em relatar este caso a xxxx. 6. Sou vítima de crime de fraude. Por que a vítima me exclui? A razão pela qual usei o cartão de crédito Chase estava sendo protegido dessa situação. 7. O especialista me disse que preciso pagar juros dessas transações de fraude por causa delas são válidas. Por que preciso pagar dinheiro por outros usos de outras pessoas? 8. Apenas algumas das transações foram validadas. Não faz sentido dizer que XXXX verificou Somes e, novamente, não preciso me importar com o que XXXX disse. Porque eu não usei nada. 9. Até eu recusei transações de fraude, não funcionará. É o problema do sistema deles. Como posso provar mais que essas transações não foram feitas por mim e pela ação do crime? Esse acordo de segurança é outro tipo de fraude para mim, porque eu estava totalmente exposto ao crime e nunca protegido. Eu não tive nenhum problema com o cartão de crédito antes que essa situação acontecesse. Novamente, sou vítima de roubo de identidade e não fiz ou autorizei essa acusação. Quero meu dinheiro de volta, e quaisquer finanças e outras cobranças relacionadas à quantia fraudulenta também serão creditadas, e que recebo uma declaração precisa.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Espero que o Chase não evite m\n",
            "======================================================\n",
            "Texto: Eu tenho um empréstimo com XXXX XXXX, XXXX XXXX XXXX e JPMCB. Eu sempre fiz meus pagamentos a tempo. Como você pode ver, eu sempre tive um registro de pagamento estelar com esta empresa. Tentei entrar em contato com XXXX, XXXX, XXXX, XXXX XXXX, XXXX XXXX XXXX e JPMCB sem resolução bem -sucedida. Definitivamente, houve um erro da parte deles.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  No último trimestre, e\n",
            "======================================================\n",
            "Texto: Eu me inscrevi em um empréstimo com o Chase Bank. Fui negado que eles dizem com base no relatório de crédito da XXXX. Eles declararam (1) inadimplência grave (2) Número de contas inadimplentes (3) contas rotativas ou rotativas muito altas (4) de tempo desde que contas inadimplentes. Isto é falso. # Xxxx registro público atrás pago. Não tenho contas inadimplentes. Não tenho saldos de crédito alto, pois o XXXX Dollares estava na minha conta Chase. Duração de tempo?????\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Apelaria da minha decis\n",
            "======================================================\n",
            "Texto: Fomos contatados pela Chase que eles suspeitavam de fraude no cartão de crédito em nosso cartão por {$ 5,00}. Verificamos as outras compras por não poderiam verificar se uma que eles declararam originados em xxxx. Nosso cartão de crédito foi imediatamente parado e eles enviaram um novo cartão de crédito para nós.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  A equipe de fraude da Chase entrou\n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, após um depósito direto de aproximadamente {$ 580,00}, eu tinha aproximadamente {$ 580,00} na minha conta de corrente de perseguição, terminando em xxxx. Em xx/xx/xxxx, paguei o pagamento mínimo de aproximadamente {$ 370,00} no meu cartão de crédito da Chase Freedom on -line através da minha conta de corrente. Após o processamento do pagamento mínimo, persegue no mesmo dia (xx/xx/xxxx) mantido {$ 150,00} em fundos e deduziu a taxa de processamento legal {$ 75,00}, levando o saldo da conta a {$ 0,00}. Quando entrei em contato, recebi uma cópia de um aviso de taxa de que eu não tinha conhecimento. A taxa expira em xx/xx/xxxx de acordo com o aviso supostamente do tribunal, mas no sistema de Chase, ele mostra fraudulentamente como expirando em xxxx. Além disso, o oficial do tribunal XXXX XXXX apresentou uma certificação fraudulenta para essa taxa, mas mais importante ainda que nenhum dinheiro fosse retirado da minha conta devido ao estatuto de NJ listado abaixo, que isenta o primeiro {$ 1000,00}. Como eu mantive menos de {$ 1000,00}, nenhum dinheiro deveria ser mantido na minha conta. Além disso, enviei uma solicitação por escrito à Chase para solicitar o dinheiro que eles estão mantendo e a taxa relacionada será devolvida. Também pedi para fechar a conta corrente, pois não consigo mais atender aos requisitos para evitar uma taxa de serviço mensal de {$ 12,00}, mas o atendimento ao cliente da Chase disse que não conseguiu ajudar nessas solicitações. Xxxx. Quantia ; Exceções bens e bens gostos, ações ou interesses em qualquer corporação e propriedade pessoal de todo tipo, não excedendo o valor, excluindo o uso de roupas, {$ 1000,00} e todos vestindo roupas, propriedade de um devedor deve ser reservado, ambos antes e após sua morte, por seu uso ou de sua família ou propriedade, e não será responsável por ser apreendido ou tomado em virtude de qualquer execução ou processo civil, seja divulgado a qualquer tribunal deste Estado. Nada aqui contido deve ser considerado ou mantido para proteger da venda sob execução ou outro processo de quaisquer mercadorias, bens ou propriedades, para a compra da qual a dívida ou a demanda pela qual a sentença na qual essa execução ou processo foi emitida deve ter sido contratada, deve ter sido contratada, deve ter sido contratada, deve ter sido contratada, deve ter sido contratada, deve ter sido contratada, deve ter sido contratada, a julgamento da qual essa execução ou processo foi emitida, terá sido contratada, deve ter sido contratada, deve ter sido contratada, a julgamento da qual essa execução ou processo foi emitida, terá sido contratada, deve ter sido contratada: ou para se inscrever no processo emitido para a cobrança de impostos ou avaliações.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: direitos à exoneração\n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, meu cartão de débito foi roubado. Alguém usou meu cartão de débito para enviar dinheiro em xxxx. O titular da conta xxxx não sou eu. É de um cara. Eu chamo xxxx e disputo a transação era uma cobrança desconhecida. XXXX Abra a reivindicação e a investigação quase 5 a 10 dias. Após a investigação, eles disseram que o pagamento não é feito por mim. Eles reembolsarão dinheiro ao meu cartão de débito, mas o reembolso ficou preso no Chase Bank. O motivo é que Chase teve que encerrar minha conta porque Chase disse que o pagamento é autorizado por mim. Não é injusto. XXXX disse que o dinheiro está pronto para reembolsar para mim, mas quando eles devolvem dinheiro, Chase recusou. Liguei para o departamento de Chase, eles me pedem para dizer a xxxx, envie -me o cheque, mas xxxx não pode fazer isso porque na política xxxx, se o pagamento não for autorizado do titular do cartão estava usando esse pagamento, então quando o dinheiro do xxxx de reembolso, eles apenas têm A única maneira de enviar dinheiro de volta ao cartão de débito foi roubada. Essa transação foi tão grande para mim. Porque é {$ 3600.00}\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  . Eu preciso desse serviç\n",
            "======================================================\n",
            "Texto: Re: Conta # xxxx Reivindicação Dear Chase Bank, seu banco recentemente descontou um cheque da minha conta (cheque # xxxx) para beneficiário xxxx xxxx.   O cheque foi enviado por correio, mas nunca recebido por xxxx xxxx, mas você descontou o cheque e removeu os fundos da minha conta, mesmo que o cheque não tenha sido endossado.   Como cliente de longa data do Chase Bank, estou horrorizado e xxxx como um banco credível pode cometer esse erro e depois informar ao cliente que sou igualmente culpado porque cito compartilhar a responsabilidade com o banco.   Você poderia facilmente devolver os fundos à minha conta enquanto conduz sua investigação, o que me disseram pode levar até 3 meses. Mas, ao não fazer isso, você eliminou qualquer confiança que tenho em você como uma instituição financeira credível.   Esteja aconselhado que eu tenha apresentado reclamações a várias agências, incluindo o CFPB (Consumer Financial Protection Bureau). Sem dúvida, você estará ouvindo deles em um futuro próximo.   Sinceramente, xxxx xxxx xxxx\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Chase Bank- Mobile App and Telephone Automated Services Fornecendo informações de conta incorretas sobre saldos e transações da conta. <P/> onxx/xx/xxxx, fiz duas transações de {$ 2000.00} e {$ 500,00} para um depósito em um carro novo com meu cartão de débito. A transação publicada na conta imediatamente e os fundos foram retirados. <p/> xx/xx/xxxx -As duas transações de {$ 2000.00} e {$ 500,00} ainda foram publicadas na minha conta. <p/> xx/xx/xxxx -As duas transações de {$ 2000.00} e {$ 500,00} ainda foram publicadas na minha conta. <P/> Na manhã de xx/xx/xxxx, verifico minha conta e duas transações não foram mais publicadas na minha conta. Não houve transação pendente para {$ 2000,00} ou {$ 500.00}. O saldo da minha conta voltou para {$ 2600.00} +/-. Liguei para a concessionária de carros e eles declararam que entrariam em contato com o banco. Para fins de segurança, transferi o dinheiro xx/xx/xxxx para minha outra conta. <P/> xx/xx/xxxx, verifiquei minha conta novamente e essas duas transações não foram publicadas na minha conta. <P/> xx/xx/xxxx, verifiquei minha conta novamente e essas duas transações não foram publicadas na minha conta. <P/> xx/xx/xxxx, verifiquei a conta de que está exagerado! Então fui em frente e transferi o dinheiro para cobrir o cheque especial. As duas transações que desapareceram finalmente publicaram 4 dias úteis após a data da transação. <P/> Liguei para o atendimento ao cliente da Chase e passei pelo banco telefônico automatizado. O serviço telefônico automatizado declarou que o saldo da minha conta era de US $ 1160,00 +/- quando no aplicativo declarou que possuía $ -2200,00 +/-. Finalmente cheguei a um representante de atendimento ao cliente e expliquei a ela o que havia acontecido. Ela continua se desculpando pela confusão como se eu fosse louca !! <P/> Perguntei a ela se essas duas transações foram postadas na minha conta? Por que ainda consegui transferir {$ 2000.00} em xx/xx/xxxx da minha conta de cheque se o dinheiro não estivesse disponível? O Chase Online Banking não permitirá que você transfira fundos entre duas conta e exagere sua conta. Além disso, como pude retirar {$ 60,00} do caixa eletrônico em xx/xx/xxxx se minha conta fosse negativa? Por que o Chase aprovaria várias transações de débito quando minha conta só puder ir {$ 400,00} no negativo antes do declínio da transação? Minha conta foi superada por {$ 2200.00}! Não é adicionado. O representante do atendimento ao cliente não me ajudaria ou reverteria as taxas de cheque especial! Ela continuou dizendo que eu peço desculpas pela confusão. <P/> Chase Bank está fraudando seu cliente !!!\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  Fornecendo informações\n",
            "======================================================\n",
            "Texto: Eu tenho tentado entrar em contato com XXXX para fechar minhas contas de verificação e poupança há mais de uma semana agora por telefone e e -mail, mas simplesmente não consigo conseguir ninguém em seu atendimento ao cliente. Meu contato inicial foi em xx/xx/2021 e eu tenho tentado acompanhar todos os dias desde então.   Eu retirei com sucesso todo o saldo de ambas as contas e eles estão agora em {$ 0,00}; portanto, não deve haver problemas para fechar nenhuma das contas.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  \n",
            "\n",
            "Também já env\n",
            "======================================================\n",
            "Texto: Usei meu cartão de crédito para reservar um hotel em xxxx. O hotel tentou me sobrecarregar quando eu realmente cheguei lá e não estava pronto para homenagear as tarifas pré-reservadas. Eu não fiquei lá, mas eles ainda me acusaram pela minha reserva. Sob meu cartão de crédito, devo obter seguro para minhas compras, mas este cartão não honra isso. Eles não reembolsarão a taxa.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "A melhor maneira de lid\n",
            "======================================================\n",
            "Texto: Peguei alguns empréstimos para estudantes particulares da Chase em XXXX para me ajudar a passar enquanto frequento a faculdade xxxx. Nunca recebi nada afirmando quando meus pagamentos deveriam começar. No xxxx xxxx, recebi um telefonema afirmando que meus empréstimos para estudantes estavam atrasados. Quando conversei com eles, expliquei que nunca recebíamos nada pelo correio sobre eles chegando ao vencimento. Eles me disseram que eu tinha que pagar todo o saldo que estava vencido naquele dia. Eu informei que não tinha {$ 1500,00} e perguntei se eu poderia simplesmente mover os meses em que estava atrasado para o final do empréstimo e começaria a pagar naquele mês. Eles disseram que isso não era possível. Perguntei se eu poderia simplesmente enviar pagamentos parciais e eles também disseram que isso não era possível que tivéssemos que pagar todo o saldo do vencimento. Deixei isso porque na época eu não tinha tanto dinheiro. Enquanto isso, descobrimos que a razão pela qual nunca recebemos nada afirmando que o devido era porque nosso co-signatário estava em falência. Dissemos a eles naquele dia no telefone que 1.) O co-assinante faleceu xxxx xxxx, xxxx e 2.) Sua falência foi arquivada em xxxx xxxx, xxxx, que foi antes de ele falecer. Isso deveria ter sido obviamente observado, já que isso foi 8 meses após a falência ter sido final e ele faleceu. Posso dizer agora que minha esposa e eu tivemos que registrar a falência em xxxx devido a contas médicas. Desde a nossa falência, nunca tivemos um pagamento atrasado em nada. Bem, começamos a procurar uma hipoteca este mês e, desde então, descobrimos que seus empréstimos para estudantes de perseguição foram carregados. XXXX NUNCA recebemos nada pelo correio ou um telefonema afirmando que eles seriam carregados e eles sabiam que, em xxxx, o co-signatário não estava mais em falência e também faleceu. Eles foram carregados xxxx xxxx, xxxx. Se tivéssemos recebido algo afirmando que eles seriam cobrados, teríamos feito o que fosse necessário para reunir o dinheiro para obter o pagamento, se isso significava ter uma venda de garagem para chegar ao dinheiro ou pegá -lo emprestado . Quando descobrimos, ligamos para Chase e eles nos transferiram para a agência de cobrança (que eu não tenho o nome deles agora). Perguntei se poderíamos começar a fazer pagamentos mensais e essa foi a resposta deles: você tem duas opções. 1.) Pague o saldo inteiro vencido hoje, que é {$ 42,00}, xxx ou 2.) Tome esse valor em 24 meses para pagá -lo, o que é de US $ xxxx por mês. Eu trago para casa {$ 2000.00} por mês do meu trabalho, como devo pagar tudo isso por {$ 200,00} por 2 anos para eles ?? Perguntei se poderíamos enviar um cheque pelo correio todos os meses por {$ 100,00} ou o que quer que pudéssemos pagar naquele mês e eles disseram: \"Você pode enviá -lo para o Chase Bank, mas isso não mudará o status e você não Receba qualquer recibo/documentação pelo correio mostrando seus pagamentos. '' Bem, eu não vou enviar dinheiro para uma empresa se não tiver garantia de que está tirando o saldo. E se tiver sido transferido sobre Para uma agência de cobrança, então por que eu enviaria o dinheiro para perseguir se estiver aparecendo no meu crédito \"carregado como dívida incorreta\" e \"lucro e perda de perda\"? Além disso, não mostra nenhuma agência de cobrança/ Os itens no meu relatório de crédito também. Se a empresa estivesse disposta a trabalhar conosco, ficaríamos felizes em fazer pagamentos. Mas me pedir para pagar os 90 % do meu pagamento por 2 anos é absolutamente ridículo.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Por favor, preciso de ajuda para\n",
            "======================================================\n",
            "Texto: Eu me inscrevi em um cartão de marca de hotel, on -line, atendido por Chase. A resposta automatizada deles foi uma delas, eles precisavam fazer pesquisas adicionais. Depois de alguns dias não recebendo uma resposta, entrei em contato com Chase. Disseram -me que meu pedido não foi aprovado desde que eu abri vários novos cartões de crédito nos últimos anos. Isso, apesar de um relacionamento de longo prazo com Chase, com uma classificação de crédito excepcional e grandes linhas de crédito com esta instituição financeira. Recebi uma carta vários dias depois, reiterando a desaprovação de Chase do meu pedido. Eu então escalei para os escritórios dos executivos. Fui contactado por XXXX, que basicamente me disse que não podia aprovar meu pedido, pelo mesmo motivo. A inflexibilidade mostrada por Chase me coloca em um dilema. Eu quero deixar de fazer negócios com eles. No entanto, se eu cancelar seus cartões, serei penalizado porque as agências de relatórios de crédito me crucificarão, com cortes drásticos nas minhas pontuações de crédito. Isso, por causa do meu corte, de grandes linhas de crédito. Parece que várias consultas de crédito também podem manchar seu histórico de crédito. Por que não tenho uma opção protegida de matar os produtos da Chase? No momento, meu único recurso é substituir essas linhas de crédito por recursos não perseguidores.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  Não é exatamente uma sol\n",
            "======================================================\n",
            "Texto: Em xx/xx/2020, cansei de fazer login on -line e isso me disse que minha conta foi bloqueada, então fui ao banco para tentar tirar dinheiro e o caixa me disse que não posso retirar dinheiro e ligar para esse número, então eu Chamado Chase Bank e eles me disseram que recebi um depósito em xxxx e eles querem prova do depósito. Eles estarão fechando minha conta e mantêm meus fundos até que eu receba provas e, se eu não receber provas, eles manterão meu dinheiro. Eles não me enviaram nenhuma carta pelo correio nem me ligaram, eles me enviaram uma bruxa on -line de notificação que eu não posso obter, porque eles bloquearam minha conta. A única informação que estou recebendo é se eu ligar e a informação é diferente toda vez que ligar.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  Eu pedi para eles me envi\n",
            "======================================================\n",
            "Texto: Saudações: Por dois períodos de cobrança (xxxx e xxxx), o Chase não aplicou pagamentos enviados ao saldo da minha conta; Como resultado, Chase relatou que estou entre 60 e 89 dias de atraso, o que não é verdadeiro. Eu estive em contato com Chase sem parar; No entanto, eles estão se recusando a corrigir o erro que fez com que minha pontuação de crédito diminua em mais de 100 pontos. Eu até criei uma planilha para mostrar que meus cálculos correspondem ao saldo da conta do meu empréstimo para xxxx, xxxx, xxxx, xxxx, xxxx e xxxx. No entanto, meus cálculos não correspondem a XXXX e XXXX porque o Chase não conseguiu deduzir pagamentos anteriores do saldo da minha conta para xxxx e xxxx.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "Prezado Sr. / Sra\n",
            "======================================================\n",
            "Texto: Acredito que meu prestador de serviços (Chase) e provedor de empréstimos (xxxx xxxx) não estão sendo transparentes e podem intencionalmente ser enganosos consumidores/mutuários em relação a seus direitos de renunciar/cancelar o PMI sob o HPA e que eles podem não estar seguindo o HPA, o que eu Encontrei especifica uma relação LTV de 80 % nos empréstimos convencionais.   Recebi uma carta/divulgação de Chase sobre a Lei de Proteção do Proprietário da casa (HPA) para mutuários com PMI, que afirma que para empréstimos fechados após xxxx, o PMI pode ser cancelado se \"1) a data em que o saldo do seu empréstimo é Primeiro programado para atingir 80 % do valor original da propriedade, ou 2) a data em que o equilíbrio de princípio atinge 80 % do valor original da propriedade ''. Após uma investigação adicional sobre o cancelamento do meu PMI em xx/xx/xxxx, Recebi uma carta de perseguir xx/xx/xxxx descrevendo condições adicionais necessárias para cancelar o PMI: - menos de 2 anos: relação LTV de 75 % ou menor com melhorias significativas - 2 a 5 anos: 75 % LTV Ratio ou menor - 5 ou 5 anos: índice LTV de 80 % ou menor que chamei Chase em xx/xx/xxxx em xxxx para coletar mais informações sobre o cancelamento do PMI e foi informado que 1) não precisava de uma avaliação em minha propriedade se fizesse um pagamento fixo de pagamento fixo ao meu princípio com base no valor atual de avaliação/avaliação ($ xxxx) com a proporção LTV em 78 %. Liguei para o Chase em XXXX e falei com XXXX, que mencionou que posso ou não precisar de uma avaliação se pague um montante fixo para que meu LTV diminuísse esses 75 % em relação ao valor original do empréstimo (US $ xxxx). Além disso, o XXXX mencionou que eu poderia tentar isso (ou seja, quantia fixa no valor original do empréstimo, com LTV inferior a 75 %) e \"veja se Chase cancelaria/renunciaria ao meu PMI '', mas que ela não tinha certeza\" é melhor para Experimente e veja o que Chase decide ''.   Fui transferido para XXXX, um supervisor do PMI, que me direcionou para as condições acima estabelecidas na carta que recebi do Chase em xx/xx/xxxx. Quando perguntei onde eu poderia encontrar informações para fundamentar isso, ele mencionou que isso pode ser encontrado em \"sites do governo\". Quando eu investigei ainda mais, já que não consegui encontrar essas especificações/condições em nenhum local do governo, ele Em seguida, me direcionou para xxxx xxxx xxxx xxxx xxxx: rescisão do seguro de hipoteca convencional (xx/xx/xxxx). que afirma: \"O critério de elegibilidade da relação LTV é atingido na data em que o saldo do empréstimo hipotecário atinge 75 % do valor original do valor original de A propriedade, se o empréstimo hipotecário for temperado por dois ou mais anos, mesmo que o prazo original especificado não tenha decorrido. '' XXXX também mencionou que a carta inicial que recebi da Chase estabelecendo a proporção de 80 % LTV era para \"empréstimos convencionais\". Então perguntei se meu empréstimo era convencional. Ele disse que era, mas que os 80 % ainda não fizeram Aplique ao meu empréstimo.   Esta orientação difere das informações que recebi 1) Durante o processo de fechamento em minha casa, 2) a carta que recebi de Chase sobre meus direitos como mutuário sob HPA, 3) Os dois telefonemas anteriores que fiz para perseguir meu empréstimo/ PMI em xx/xx/xxxx e 4) Todas e quaisquer informações de código aberto que eu pude encontrar no CFPB e nos sites do Federal Reserve Board, estabelecendo as condições do HPA.   Eu, como consumidor, devo ser informado das condições necessárias para cancelar o PMI no HPA antes de aceitar os termos do empréstimo específico que se aplica a mim (por exemplo, convencional). Depois de passar por toda a minha papelada de fechamento, só consegui encontrar informações que afirmam que o PMI pode ser cancelado após atingir 80 % de LTV. Isso é confuso para os consumidores, na melhor das hipóteses, e intencionalmente enganosos no seu pior. Pessoalmente, gostaria de pagar uma quantia fixa agora para obter meu empréstimo a 80 % LTV, cancelando meu PMI, sem ter que pagar por uma avaliação ({$ 510,00}), que é configurada por um servicador em que não confio (Chase ), com base nas orientações de um provedor de empréstimos (XXXX), que três agentes da Chase não conseguiram interpretar. Se esses agentes de atendimento ao cliente não conseguem nem entender a orientação, que esperança existe para os consumidores cotidianos entenderem.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Recebi um e -mail no XXXX/XXXX/2015 dizendo que o Chase Bank experimentou alguma fraude em potencial e, como precaução, eles estavam me enviando um novo cartão. Enquanto esperava o novo cartão, continuei usando o meu antigo. No XXXX/XXXX/2015, fui retirar dinheiro do caixa eletrônico para pagar meu aluguel e o caixa eletrônico me levou a ligar para o banco em que eu fiz. Fui informado de que o cartão havia sido desativado e eu já deveria ter recebido meu novo cartão. O agente que eu estava conversando verificou meu endereço e me aconselhou a verificar minha caixa de correio. Xxxx que eu faço regularmente) depois de verificar minha caixa de correio e não encontrar o cartão lá, voltei para o banco e fiz um caixa retirado {$ 400,00} da minha conta. Fui então para casa e liguei para o banco novamente e aconselhei a situação em que esse agente me pediu para verificar também meu endereço e disse que o endereço errado estava registrado. Aconselhei o agente que meu endereço deveria ter sido o mesmo endereço que tive na minha conta nos últimos 2 anos. Também aconselhei que acabei de falar com um agente anteriormente que confirmou meu endereço por meio da verificação. O agente atualizou meu endereço e aconselhou que \"entregaria\" meu cartão para mim, o que levaria 48 horas. Agora é xxxx/xxxx/2015 e não recebi nada. Trabalho de segunda a sexta -feira xxxx a xxxx. I raramente tenho tempo durante a semana para transformar em uma filial para retirar dinheiro. Tenho 4 cheques que preciso depositar em minha conta, mas b \\ c desse inconveniente não foram B \\ c Não tenho acesso ATM e tenho sido Forçado a depositá -los em minhas namoradas que a conta corrente. 65 % do meu salário é depositado nessa conta tão desnecessário dizer que essa provação foi um dos pior inconvenientes com os quais lidei de qualquer banco. Sou pago sexta -feira xxxx/xxxx/2015 e terá que sair do meu caminho para entrar em um ramo para retirar a maior parte do meu dinheiro para não voltar. Sou um homem com pequena paciência e fechei minha conta xxxx xxxx xxxx por inconvenientes como tal e embora eu não possa ser Valorizado como cliente aqui, tenho certeza de que outro banco me valorizará.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  Não tenho dinheiro suficient\n",
            "======================================================\n",
            "Texto: A quem possa interessar, enviei -lhe uma reclamação três vezes. Pedi respostas para minha pergunta ainda não recebi essas respostas da Chase. Quando alguém aplica dinheiro incorreto e não explicará por que ou onde está, não é essa fraude. Bem, eu preciso responder a essas perguntas e Chase está mentindo para você dizendo que eles responderam a essas perguntas porque elas não têm 1. Enviei cópias de recibos mostrando que paguei mais em garantia que eles ainda precisam contar onde está o dinheiro. Eu quero que seja reembolsado ou colocado em garantia. Eu pago no banco é por que tenho cópias e prova do que foi pago e quanto eles me disseram que não aceita dinheiro extra por custódia, mas o stub de pagamento tem a opção de fazer assim.   2. Segure uma cópia dos cheques que escrevi que existem 31 cheques que eles mostram sobre a história que são uma quantia, mas, na realidade, há mais dinheiro pago que eles mostram menos. Exemplo abaixo: xxxx escreveu um cheque xxxx que era xxxx, xxxx extra para ir em custódia que eles mostram em seu valor de verificação de registros foi 914.93. Existem 31 pagamentos como esse e nenhum dinheiro extra foi aplicado ao garantia. Não é essa fraude.   3.On xxxx Eu escrevi um cheque para xxxx, não mostra que não foi aplicado. Número de verificação Número xxxx. Onde está o dinheiro 4. Mostro na folha de transações de história, eles me enviaram que não havia dinheiro colocado em garantia nos meses de xxxx, xxxx, xxxx e xxxx xxxx.   5. Eu não mostro que não foi adicionado ao meu garantia desde xxxx/xxxx/xxxx.   6. Quando eles reverteram, ainda nunca entraram na quantidade correta.   7. Em cópias, eles me enviaram, há muitos pagamentos não aplicados à minha conta. Exemplo abaixo de XXXX/XXXX/XXXX e XXXX/XXXX/XXXX Ambos os pagamentos mostram suspense e nunca aplicados e, na cópia, eles me enviaram, mostra -se não aplicado à direita . Então eles têm muito dinheiro listados como redução e não aplicados. E listados como supsência 8. Eles compraram Washington mutal e minha pergunta a eles é por que minhas declarações dizem o garantia e o princípio, mas nenhum dinheiro foi aplicado ao princípio do empréstimo tem essas cópias também.   Sinto que algo está acontecendo com meus pagamentos e com o dinheiro que dei a eles, quero algumas respostas, por favor. Eles dizem que eles enviaram respostas que não responderam a todas as perguntas. Eles fizeram um mod na minha casa e foi quando minha casa estava debaixo d'água e nunca ofereceram nenhum dos programas lá fora do pagamento. Os mutuários que enviaram um formulário de reclamação de pagamento válido através do XXXX receberam um cheque de aproximadamente {$ 1400,00}. As verificações foram divulgadas entre xxxx xxxx, xxxx e xxxx xxxx, xxxx. ( The initial deadline to make a claim was XXXX XXXX, XXXX. ) Refinancing for Underwater Homeowners XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX XXXX\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  XXXX XXXX XXXX XXXX XXXX\n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, paguei o saldo devido à perseguição à liberdade acct # - xxxx. Devido às taxas cobradas na minha conta, elas não mostraram um saldo XXXX até xx/xx/xxxx. Eu descartei os cartões e nunca mais usei a conta. Continuei a receber ofertas da empresa que os descartei também. Em xx/xx/xxxx, comecei a receber chamadas de uma agência de cobrança para a qual expliquei que essa conta havia sido paga desde xx/xx/xxxx. A agência de cobrança tentou entrar em contato com Chase e não recebeu resposta. Em xx/xx/xxxx, outra agência de cobrança entrou em contato comigo e eu disse a eles a mesma coisa. Em xx/xx/xxxx, escrevi uma carta de Chase uma carta informando todos os fatos e eles responderam enviando 75 envelopes, todos com as mesmas páginas exatas nelas. Em xx/xx/xxxx, pude alcançar um representante do Chase e estava ao telefone por um longo período de tempo porque o Chase mudou o número da conta e eles não conseguiram localizar a conta. Fui então informado sobre cobranças usando xxxx a xxxx, xxxx tudo dentro de um período de 4 dias. Não tenho uma conta XXXX e não fiz essas cobranças. Chase Freedom continua a relatar negativo em meus relatórios de crédito. O Chase precisa corrigir essas cobranças fraudulentas e mostrar um saldo XXXX.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Atualmente, estou em um empréstimo que não foi realmente divulgado adequadamente pelo meu credor e agora tenho tentado encontrar alguém alguém para me ajudar a obter essa alteração de empréstimo em 30 anos fixo e ter uma conta de apreensão anexada. Tentei modificações de empréstimo apenas para me dizer, sou atualizado no pagamento da casa e tenho muito patrimônio em minha casa. Foi -me dito por um gerente da Chase devido ao BK que eles não ajudam e não me colocaram em impacto. Minha casa não foi incluída no BK e eu só preciso que esse empréstimo seja transformado em um xxxx fixo com o que a afetar e a uma taxa acessível pode gerenciar. Eu sinto que Chase não foi útil nessa situação, então espero que você tenha que servir para ajudar. obrigada\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: . \n",
            "\n",
            "Eu entendo sua\n",
            "======================================================\n",
            "Texto: Fiz uma modificação de empréstimo com o Chase Bank sob o Programa Federal Hamp no XXXX de XXXX, durante esse período, tive dificuldades financeiras e fui aprovado para a modificação do empréstimo. Durante a assinatura dos documentos, notei uma cláusula de um valor de um balão {$ 210000.00} na data de vencimento, perguntei ao processador sobre a cláusula e as respostas foram, para não se preocupar, esse foi o acúmulo de interesse durante a vida da vida do empréstimo. Durante os anos seguintes, liguei para o banco para fazer as mesmas perguntas sobre esse valor de balão e a resposta sempre foi a mesma, para não se preocupar, esse é o interesse acumulado pela vida do empréstimo. Recentemente, recebi uma oferta do Chase Bank para reformular o empréstimo, o banco enviou documentos para eu assinar, notei a mesma cláusula, mas agora com notas adicionais (anexou os documentos de reformulação). Na minha curiosidade, liguei para o Banco do Chase para perguntar sobre essa cláusula, mas desta vez eles informaram que esse era um valor que eu precisava pagar se fosse vender ou pagar o imóvel. Minha queixa é a seguinte, procurei o Bank para obter ajuda sob o programa federal Hamp para salvar minha propriedade, em vez de me ajudar, eles me enganam, com malícia e propósito. Desde o primeiro dia, seu objetivo era fraudar e ganhar com um proprietário de casa em dificuldades. Eu cheguei com uma dívida de {$ 360000.00} e saí com uma dívida de {$ 530000.00} apenas assinando este contrato. O Chase Bank não me ajudou nada, em vez disso, eles me fraudaram como fizeram com milhares de clientes. Peço ao governo federal que coloque esses ladrões gananciosos na prisão e faça com que eles paguem por todo o sofrimento e estresse que causaram às pessoas honoráveis ​​deste grande país.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Abri uma conta de cobrança através do Slate Chase Bank anos atrás (eu acho que há mais de 7 anos), cheguei atrasado com pagamentos e fui contatado por coleções sobre o fechamento da minha conta e a criação de acordos para evitar ações legais por telefone ou ameaçado de tomar eu para o tribunal. O saldo estava sob {$ 2000,00}, acredito que possivelmente menos, não era extremamente alto. Foi -me dito para interromper a ação legal que eu precisava celebrar um contrato de pagamento para pagar o saldo, a chamada foi registrada. O contrato era que eu configurei pagamentos mensais automáticos de {$ 39,00} para pagar o saldo. A conta de crédito foi fechada e não pôde ser usada até que o saldo fosse pago integralmente. Nenhuma acusação de juros ocorreu porque a conta foi fechada e na cobrança. Eu concordei. Desde então, tive {$ 39,00} por mês retirado da minha conta bancária, mas eles continuaram cobrando juros todos os meses, portanto, com a conta encerrar todos os anos de pagamentos, meu saldo ainda é {$ 1500.00}. Tentei ligar e sou recusado a falar com um gerente, o atendimento ao cliente diz que a conta está fechada e eles não falarão comigo sobre a conta até que o saldo seja pago integralmente. Eu acho que essa foi uma atividade de coleta fraudulenta no começo e ainda hoje. Gostaria de parar de receber contas e acredito que eles me devem pagar em excesso ao saldo original que paguei. Também registrei uma queixa no XXXX e ao escritório de advogados gerais hoje.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Será possível obter algum\n",
            "======================================================\n",
            "Texto: Minha instituição financeira é perseguir cliente particular. Depositei um xxxx xx/xx/xxxx, o Chase não pode localizar esse depósito violando vários regulamentos de impostos bancários. Iniciei a regressão de pagamentos mensais a serem feitos da minha conta Chase CPC para xxxx xxxx xxxx no 3º depois que meu cheque xxxx foi depositado direto e o Chase Rotineiramente falhou em homenagear esses pagamentos e, ao fazer isso, minha conta com xxxx xxxx foi deixada em conferência . Esses pagamentos foram deixados em um status \"pendente\" que os fundos foram retirados da minha conta, mas não pagos ao comerciante, deixando o comerciante não remunerado e que levou a um efeito de bola de neve em taxas de juros agitadas. Meus pagamentos estavam sendo aplicados aos juros E não o princípio me deixando de cabeça para baixo no empréstimo e uma marca negativa no meu relatório de crédito e em risco de perder meu veículo. Isso está acontecendo hoje também. Em xx/xx/xxxx, comprei um computador online com xxxx. O pagamento de {$ 590,00} a xxxx permaneceu em um status \"pendente\" desde então. XXXX não enviou meu computador porque eles afirmam que Chase recusou a transação e Chase afirma que o xxxx não processou a ordem, independentemente Libere os fundos de volta à minha conta e o XXXX não enviará meu computador. Chase está violando o Regulamento E sob sua regra e precisa investigar.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Além disso, por conta dos\n",
            "======================================================\n",
            "Texto: Essa reclamação é para um pagamento duplicado incorreto dos meus impostos sobre o Devocro da Propriedade Imobiliária, que meu servidor de hipotecas, Chase, não conseguiu correr imediatamente. Meu número de empréstimo do Chase é xxxx e meus impostos sobre a propriedade imobiliária são pagos ao XXXX County, Ohio. Meu número de encomendas do meu XXXX County, Ohio é xxxx.   Xxxx xxxx xxxx xxxx originou minha primeira hipoteca convencional em xx/xx/xxxx. XXXX XXXX Atendeu minha hipoteca em nome de xxxx xxxx através de xxxx, xxxx. Em ou sobre xx/xx/xxxx, xxxx pagou meus impostos imobiliários de {$ 2500,00}. Este pagamento foi pela segunda metade do xxxx, devido a xx/xx/xxxx.   Xxxx xxxx xxxx xxxx vendeu a manutenção da minha hipoteca para perseguir, em vigor ou aproximadamente xx/xx/xxxx. Em ou sobre xx/xx/xxxx, o Chase fez um pagamento de imposto duplicado ao Condado de XXXX, Ohio, pelo mesmo valor de {$ 2500,00} que o XXXX já havia pago em xx/xx/xxxx. Esse pagamento duplicado fez com que meu saldo de garantia de Chase fosse negativo. Descobri esse pagamento duplicado quando recebi a declaração de hipoteca por correio da Chase, datada de xx/xx/xxxx. Relatei o erro para perseguir o telefone xx/xx/xxxx.   Quando liguei para Chase, fui transferido para a unidade de serviço tributário. Parece que o Chase não conseguiu carregar meu histórico de contas de custódia do meu ex -funcionário em seu próprio sistema, porque eles não tinham registro do pagamento de imposto XXXX feito xx/xx/xxxx. Em vez disso, Chase disse que era meu trabalho provar a eles que o pagamento deles era uma duplicata.   Eu tentei enviar uma impressão por fax do site do meu ex -Servicer para perseguir, mas o número de fax que Chase me deu não funciona com a máquina de fax no meu escritório. Chase não me deu nenhum endereço de e -mail para enviar uma imagem em PDF da impressão do meu ex -servicer. Nesse ponto, sou forçado a enviar uma cópia da impressão para perseguir, às minhas próprias custas.   Estou preocupado com o fato de que, mesmo que o Chase finalmente perceba que eles fizeram um pagamento duplicado incorreto e o recuperam do Condado de XXXX, eles mostrarão uma escassez de garantia em minha conta e, portanto, aumentarão meu valor mensal de pagamento de custódia.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:   \n",
            "\n",
            "Estou solicitando uma res\n",
            "======================================================\n",
            "Texto: Fiz para XXXX XXXX Chase Branch, ramificação # xxxx. O nome do caixa é xxxx no Chase XXXX XXXX. O tempo em que entrei foi xxxx contava o dinheiro a ser depositado em um cartão de crédito como pagamento e era {$ 6000,00} quando cheguei e entreguei ao caixa o dinheiro que ela o alimentava através da máquina e dizia {$ 5000,00}. Eu me senti pressionado e ansioso, pois não quero contar o dinheiro no saguão de um banco. Ela não contava com a mão. Ela só alimentou através de uma máquina. Não pude ver fisicamente isso porque o contador é muito alto. E há um vidro à prova de balas.   Se um {$ 1000,00} desapareceu, deve estar em minha casa, pois eu moro em xxxx quarteirões de distância e foi diretamente de minha casa para o banco.   Quando saí do banco para verificar minha casa e minha casa estava vazia, voltei prontamente. Depois de falar com o gerente do banco, ela pediu desculpas. Relatamos e a gaveta está completa e me deu uma descrição do depósito. Quando perguntei se havia câmeras, ela disse que não. Quando perguntei se há uma câmera que observa o caixa, ela disse que não. Quando pedi para ver a fita, ela disse que não e afirmou que seus caixas não roubam. Quando pressionada ainda mais sobre qual objetivo a câmera serve para mostrar transações de cliente para caixa, mas nada é mostrado para proteger o cliente de um caixa roubar, pois não há fita. Eu deixei esse ponto claro para ela.   Isso não parece correto. A maioria das filiais que eu já estive em treinar câmeras no cliente e caixa mostrando a atividade do caixa. Ela me disse que seus caixa não roubam, etc. etc. Mas como isso pode proteger o consumidor? Se não houver câmeras no caixa, o caixa poderá dizer o que ela quer e colocar na gaveta de dinheiro qualquer quantia que ela queira. Recuso -me a acreditar nisso e exijo uma investigação sobre isso.   Eu refutari essa reivindicação, pois as notas de dólar xxxx não desaparecem por conta própria. Sinto como se esse caixa mal enterrasse meu dinheiro.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  Quero ver a fita de víde\n",
            "======================================================\n",
            "Texto: Caro senhor, estou no Chase Bank há 10 anos. Ultimamente, tentei conectar algum dinheiro ao meu pai em xxxx. A transação foi feita no XXXX e é transferida de transferência de fio e foi feita pessoalmente no Chase Bankxxxx XXXX XXXX, TX XXXX. E a caixa do banco conseguiu determinar o número da conta do meu pai e ela me disse que o dinheiro estará disponível no XXXX. Em xxxx/xxxx/2015, fui contactado por telefone alguém chamado xxxx: xxxx Ele quer mais informações sobre mim e meu pai e ele disse que não envia o dinheiro se eu não o fizer! Quando estou em minha conta bancária on -line, vejo que o dinheiro foi transferido até o XX/XX/2015, vi que o dinheiro foi creditado de volta sem o {$ 45,00}. Então eu pedi por esse motivo e recuperei. Perguntei ao caixa que tipo de informação você precisa para que isso aconteça, então ela disse que as duas contas bancárias é tudo. Eu não sei por que fui maltratado assim? Perguntei a todos os meus amigos que ninguém reclama! Por favor, mantenha esses grandes bancos cooperados na fila. Obrigado pela leitura. Atenciosamente.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Eu estava envolvido em um golpe militar. Os golpistas deveriam estar pagando dinheiro que eu teria adiantamento em dinheiro e colocaria uma \"conta do agente\". Os pagamentos estavam sendo retirados. Eu não tinha conhecimento. Eu conciava para ver se elegível para adiantamento em dinheiro. SA ID sim em cada ocasião. Recebi ligações do departamento de segurança várias vezes. Liguei de volta, eles perguntaram se eu fiz uma compra, eu disse que sim. Ninguém em todos se essas ocasiões me disseram que o pagamento foi retirado. Eles finalmente enviaram cartas Dizer que não poderia chegar por telefone e eu havia falado várias vezes. Entrei em contato depois que percebi que estava envolvido em golpes. Pagamentos feitos por scammer e conta deveriam estar fechados e não foram feitas mais transações. Isso w como xxx -17 Pagamento final feito em xxxx -17. I Um ganho chamado xxxx -17 para informar a Chase que o scammer tinha conta lá em nome do banco xxxx xxxx e fornecia a conta #. Eu queria dar a eles golpistas, então a pessoa de segurança nem queria lidar comigo ou tome informações. Em xxxx -17, a fraude Mers recuou todos os pagamentos depois que a conta deveria estar fechada e segura. Eu chamo xxxx -17. Eles não ouvimos ou tentamos ajudar. O {$ 24000.00} aumentou para {$ 31000.00}. Se eles tivessem me dito em alguma das inúmeras conversas telefônicas que eu tinha anteriormente que os pagamentos sendo retirados, eu teria percebido mais cedo e não estaria fora se o controle. Liguei novamente para xxxx -17, eu escussi em situação completa e perguntei por que eles não protegeram minha conta como deveriam, eles apenas preencheram a queixa e disseram que acessem o tribunal de pequenas reivindicações. Os golpistas estão na Califórnia. Eles não aceitaram responsabilidade por não me levar a sério quando eu entrei antes. Eles tinham seu dinheiro. Eu tinha chamado XXXX porque havia informado a mesma coisa. Eles me garantiram que não tenhamos o pagamento recuado porque eu havia informado sobre fraude, assim como fiz por Chase. Eles travaram a conta e a frente para sua sede. Por que Chase foi tão negligente com minha conta. Percebi que cometi erro e tentei corrigir com eles antes que eles perdessem o dinheiro novamente. Eles não conseguiram tomar as medidas adequadas depois que continuei informando -as sobre golpes para protegê -los e a mim e aquele golpista usando o banco para enganar as pessoas. Eles basicamente me surpreenderam ... onde você está para obter ajuda se entrar em contato com o banco onde há problema e eles não fazem absolutamente nada. Isso está errado, especialmente quando conversei com um funcionário em xxxx -17 e me disse basicamente que estava em segurança agora que paga e fechada. Por favor, ajude isso está acontecendo com muitas pessoas. Quando fiz o depósito em \"agente\" para os scammers, o contador disse que muitas pessoas estão depositando muito se o dinheiro nessa conta. Eu tentei repetidamente dizer isso e elas não fazem nada.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Banco com o Chase, aposento com eles há mais de 15 anos. A cada poucos meses, eles mantêm transações e cobram taxas insuficientes de fundos. Eu tenho bancos móveis, então estou constantemente olhando para minha conta. Eu tenho alguns pagamentos que saem diretamente da minha conta para que eu saiba que horas e quando eles serão elaborados. Eu também tenho depósito direto. Então eu sei quando as contas devem ser elaboradas. Depois de falar com um representante rude do cliente. Eles me informaram que você só recebe reversões xxxx por ano e eu havia atingido meu limite. Depois de ter esse mesmo problema no início do ano. O JP Morgan Chase, como todos os outros bancos, está aproveitando os clientes. Mostrou, no meu lado, que meus rascunhos haviam liberado e, no dia seguinte, recebo taxas insuficientes do fundo. Ninguém tem centenas de dólares para dar a grandes bancos que estão arrancando as pessoas. Se eu estava escrevendo cheques ruins, então mereço. Mas sou uma mulher pobre da classe trabalhadora, e o cavalheiro xxxx com quem falei poderia descuidado, sua principal preocupação era colocar dinheiro de volta aos bolsos dos bancos. Esta é uma farsa que eles fazem todos os meses a todos. Quando você precisa dar conta de cada dólar, você sabe quanto sai e o que entra. O JP Morgan Chase precisa ser investigado para cobrança e atividade ilegal.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: During ongoing review of my mortgage modification request with   XXXX   XXXX   XXXX   XXXX   XXXX  , on   XXXX   XXXX   XXXX  , I was told by  XXXX  that the owner of my mortgage note is   XXXX   XXXX   XXXX   XXXX    ( the  same company fined by the United States {$120.00}  XXXX em xxxx por violações das violações da Lei de Reivindicações Falsas) e que elas foram feitas pela oferta de modificação feita a mim. Entrei em contato com o XXXX CLIENTE AVERVICE no xxxx xxxx xxxx para solicitar uma cópia da minha nota de hipoteca e informações relacionadas à sua participação, e me disseram que o negócio de hipoteca em casa do XXXX havia sido vendido há vários anos e a única maneira de chegar a alguém Para informações adicionais, foi por correio postal. Histórico de modificação da hipoteca com xxxx xxxx xxxx - atuando em nome do xxxx xxxx xxxx xxxx: em xxxx xxxx depois de ser demitido do meu trabalho e posteriormente experimentando mais de seis metros de desemprego, iniciei o processo de explorar opções para uma modificação de empréstimo. Era explicitamente dado d a xxxx xxxx que a intenção era de fazer a nossa casa. Essa intenção foi recebida inicialmente com uma resposta positiva do XXXX XXX, comprometendo que eles trabalhariam conosco para reduzir nosso pagamento mensal. No começo, um gerente XXXX ASSE T Wored comigo para se qualificar para um pagamento mensal reduzido durante meu período de desemprego. Pagamos nosso valor mensal de hipoteca a cada mês, de acordo com este Contrato. Tendo recuperado o emprego no XXXX XXXX a uma taxa de pagamento que eu estava fazendo anteriormente, nosso objetivo contínuo de reduzir nosso pagamento mensal de hipoteca e manter nossa casa foi claramente declarada durante todo o processo de revisão com XXXX XXXX e acumulou -se pela equipe XXXX. Nesse ponto (xxxx xxxx), tínhamos enviado toda a papelada necessária e meu gerenciador de ativos XXXX atribuído inseriu -me o pagamento tonot em nossas hipotecas, esperava -se que uma modificação fosse finalizada em 30 dias. Nos três meses seguintes, representantes do XXXX XXXX começaram a solicitar informações que já tínhamos enviado em alguns casos solicitando as mesmas informações 2, 3 e 4 vezes entre xxxx e xxxx xxxx. Nós reenviamos meticulosamente informações, incluindo extratos bancários, registros de impostos, cálculos de despesas e muito mais. Entre outros pedidos estranhos que fizeram parecer que estavam adiando propositalmente nossa modificação, fomos solicitados a \"criar\" uma demonstração de lucro e perda de renda pessoal apenas para saber que eles não precisavam disso. Em mais de uma ocasião , eles alegaram que não podiam entender nossos registros de impostos (que haviam sido aceitos pelo IRS) e nós e tivemos que colocá -los em contato com nosso preparador de impostos. Fiquei surpreso quando XXXX me informou na sexta -feira, xxxx xxxx xxxx (seguindo8 Meses de coleta e análise de nossas informações financeiras) que sua oferta foi acrescentar nosso morto de forma hipotecária apenas por {$ 380,00}, no processo me parabenizando pela aprovação de nossa modificação e me avisando que a execução hipotecária era uma opção. Depois de ameaçar a ação legal, xxxx solicitado ainda Informações financeiras pessoais adicionais. Após a revisão, eles voltaram com uma oferta para reduzir nosso pagamento mensal em {$ 140,00}, juntamente com os requisitos que nosso termo de empréstimo deve ser estendido, o patrimônio líquido seja perdido, e taxas e encargos relacionados à modificação são adicionados ao valor da nossa hipoteca, comunicando novamente que a execução hipotecária era uma opção. Esta oferta de modificação foi recebida xxxx xxxx xxxx, 13 meses após a divulgação inicial. Em xxxx xxxx xxxx I Iniciei uma solicitação ao Massachusetts xxxx xxxx xxxx xxxx para investigar xxxx xxxx xxxx por fraude. É mais do que evidente que xxxx xxxx xxxx transformou uma situação de dificuldade em uma oportunidade para eles promoverem a execução duma hipoteca em seu benefício. O processo de adiar essa modificação, não cumprindo promessas verbais, movendo a equipe dentro e fora da minha conta várias vezes e o resultado geral do banco que ganha dinheiro com a transação deixa claro que todo o processo foi acalulado e fraudulento.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Este é um segundo acompanhamento da minha reclamação original. Minha reclamação original declarou isso; Em XXXX XXXX e XXXX, 2016 Tornei -me vítima de roubo de identidade no JP Morgan Chase Bank, quando aproximadamente {$ 12000,00} foi retirado das minhas contas de poupança e verificação. Até o momento, o banco reembolsa aproximadamente {$ 9500,00}. O banco se recusa a reembolsar o saldo do dinheiro recebido, que é {$ 2500,00}.   Meu 1º acompanhamento afirmou que, em xxxx xxxx, 2016, o Banco Chase creditou minha conta poupança com um total de {$ 1500,00}. Continua a haver um saldo excelente de {$ 1000,00} devido a mim. Eu tentei ligar para XXXX XXXX XXXX, Escritório Executivo, Chase Bank na XXXX Extension XXXX e ele não me liga de volta. Não tenho outro recurso que o notifique na esperança de que você seja mais uma vez intercedado em meu nome.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  \n",
            "Eu aprecio muito\n",
            "======================================================\n",
            "Texto: Eu tenho um cartão de crédito através do Chase Bank. Eu tenho três saldos diferentes no cartão, cada um com suas próprias taxas de juros. Fiz o pagamento mínimo e depois fiz dois outros pagamentos, esperando que eles fossem para o saldo com a maior taxa de juros (adiantamento em dinheiro). Não. Quando liguei, perguntei a eles sobre isso e eles disseram que não é assim que o sistema deles funciona. Disseram -me que a Lei de Responsabilidade, Responsabilidade e Divulgação do cartão de crédito de 2009 forçaria as empresas de cartão de crédito a aplicar pagamentos em excesso que o ciclo de Blling ao saldo com a maior taxa de juros primeiro. Você pode me ajudar com esse problema ou estou entendendo mal como o processo funciona.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "Eu entendo que entender ex\n",
            "======================================================\n",
            "Texto: Número do cliente xxxx - emitir refinanciamento a uma taxa baixa. A taxa atual de 5,6 % bloqueou o refinanciamento da minha casa. No ano passado, foi vendido e imobiliário de investimento e pagou xxxx em dívidas. Minha pontuação de créditos foi em xxxx. No último ano, paguei todas as contas a tempo. Veja meu relatório de crédito. Minha pontuação de crédito está em xxxx. Sem motivo. XXXX disse que a inadimplência séria nas contas pagas em 2011.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Nossa equipe criou\n",
            "======================================================\n",
            "Texto: Um cartão de crédito XXXX foi aberto sem a minha permissão. Entrei em contato com eles, eles disseram 60 dias para resolver isso para remover do meu relatório de crédito. Nada ainda\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  foi feito.\n",
            "\n",
            "A primeira\n",
            "======================================================\n",
            "Texto: Chase Fraudulento Encaminhado em mim, afirmando que falhei em meus pagamentos. Chase recusou todos os meus pagamentos mensais de xx/xx/xxxx a xxxx xx/xx/xxxx dizendo que eles não poderiam aceitar meus pagamentos enquanto estavam no empréstimo. A disputa de notificação da demissão da modificação do empréstimo # xxxx e/orreInstatement da modificação do empréstimo # xxxx. \"Chase não respondeu ao meu\" aviso para disputar a demissão e//orreinstatement et al \"até depois de me encerrarem. Chase ofereceu a \"modificação de empréstimo demitido # xxxx '' para mim novamente (com as figuras erradas intactas) depois que elas foram necessárias para mim. incluindo a Lei de Reforma e Proteção ao Consumidor de Wall Street e Dodd-Frank. Explique ao proprietário de como o Opserivou ao adicionar mais 14 anos ao seu empréstimo quando eu não estava para trás em Principal & Interes. Derivado adicionando 14 anos quando o proprietário não estava atrasado em diretor e interesse. Chase me disse que a custódia não tinha permissão para fazer contas até o final do empréstimo, portanto, o proprietário queria saber exatamente o que eles acrescentaram ao empréstimo ao seu empréstimo, com 14 anos adicionais em seu empréstimo fixo de 30 anos. Se eu não estivesse atrasado e até estava por trás de um (1) mês que teria calculado aprox. {$ 460,00} ASSEMA MINHA NOTA MENTARGA DE MORTEGEM MENSAL NO MOMO ASESCROW não havia sido incluído na minha nota mensal até que a modificação do empréstimo tenha sido concluída. Eu fui consumido para fazer uma panela de garantia adicional de quantia fixo para perseguir. Então, como eles chegaram a 14 mais anos adicionados ao empréstimo? A fraude de Fearsaddicional do proprietário foi feita com seu empréstimo em empréstimo # xxxx (xx/xx/xxxx) também. Novamente, por que ChaseRefusing está substituindo seus dados/documentos de xx/xx/xxxx a xx/xx/xxxx muito menos os dados/documentos xx/xx/xxxx a xx/xx/xxxx para acesso do proprietário. Eles mantiveram seus dados e documentos de suas contas desde o fechamento de seu empréstimo em xx/xx/xxxx.homeowner e seu conselheiro habitacional não tinham acesso aos seus dados de conta/documentos para trabalhar para modificação de empréstimos # xxxx & chase se recusou a devolver dados /Documentos através do presente dia. Eles agora removeram todos os seus dados/documentos de sua conta, para que ela não tenha acesso a tudo. Chase deixou apenas o nome dela, que também é incorreto, pois não mostra seu último nome legal. Os escritórios executivos se recusaram a alterar os números de retecção submetidos aos subscritores que afirmavam que os cálculos do subscritor estavam corretos. Sim, os cálculos estavam corretos, mas os números dados e usados ​​pelos subscritores XXXX (funcionários do JP Morgan Chasepaid) foram dados a eles pela escroanálise de Chase, que estava incorreta, pois não refletia os créditos adequados devido ao proprietário desde que ela pagou xx/xx/xx/xx/ XXXXSCROW em montante fixo em xx/xx/xxxx e continuou a pagar demais pagamentos, incluindo uma nota de hipoteca mensal de garantia adicional. Chase está sendo \"sarcástico\" por si só, continuando apoiando os figuras dos subscritores corretos conforme calculado, mas se eles tivessem recebido as figuras de custódia correta, isso não teria sido anizado para o proprietário. Mas Chase se recusou a dar aos subscritores as figuras corretas, executando facilmente novamente Análise como xxxx xxxx planejada. A análise de escrota usada foi xx/xx/xxxx e a temodificação não foi feita até xx/xx/xxxx. Seeletter para perseguir atty xxxx xxxx, aviso para contestar a modificação # xxx, resumo de ações, mensagens mensais\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: /informações em ar\n",
            "======================================================\n",
            "Texto: ONXX/XX/XXXX Um Pay Chase Autorizado (xxxx) removido {$ 400.00} da minha conta de corrente Chase. Aconselhei a Chase imediatamente que não autentive essa transação. Chase me aconselhou que eu precisaria assinar uma declaração de disputa. Fui ao ramo dentro de uma hora assinei a declaração e enviei por fax, conforme solicitado por Chase. Chase me aconselhou e ao vice -presidente da filial que um crédito seria emitido para minha conta por {$ 400,00}. O crédito apareceu na minha conta de corrente congelada em xx/xx/xxxx. Passei a verificar minha conta onxx/xx/xxxx para ver o crédito removido. Depois de passar várias horas no telefone com Chase, eles me aconselharam que era um crédito de emergência. Não entendo como você pode emitir um crédito e removê -lo. Este é o meu dinheiro. Ninguém me ligou de volta e todo mundo me dá histórias diferentes.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Não tenho nenhuma explicaç\n",
            "======================================================\n",
            "Texto: Descoberto recentemente, eu estava sendo cobrado mensalmente {$ 11,00} por um cartão de crédito xxxx xxxx não autorizado do Chase Bank de uma conta bancária Chase que foi fechada no xxxx xxxx, xxxx. No xxxx xxxx, xxxx, entrei em contato com o Chase e os informei que não autorizei a placa xxxx xxxx e não recebi um cartão ou ativei o mesmo. Fui aconselhado por Chase nessa data de que eles fechariam o cartão XXX XXXX como um possível problema de fraude. Em xxxx xxxx, o serviço XXXX Chase CardMember enviou uma carta aconselhando que a conta do cartão de crédito já venceu no valor de {$ 35,00}. Incluindo o valor vencido e o pagamento atual devido. \"Falha em pagar esse valor pode resultar na sua conta estar fechada. '' Em ​​xxxx xxxx, xxxx minha pontuação de crédito caiu de xxxx para xxxx, uma queda de ponto xxxx, por site xxxx xxxx xxxx. Aconselhamento do Serviço por e-mails Não houve saldos pendentes e \"o cliente não estava ciente dos encargos de associação principal. '' No XXXX XXXX, XXXX CARTMEMBER SERVIÇOS DE MEMBRO DE CASA ALVOGADOS ATRAVÉS DE UMA CARTA RELATIVA EM TERMINADO NO XXXX, \"Fechamos esta conta para impedir a fraude. ''\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: Concluímos que esta conta\n",
            "======================================================\n",
            "Texto: No XX/XX/XXXX, conversei com o gerente de vendas para ser pré -qualificado com xxxx xxxx xxxx xxxx, estou completamente ciente de que, durante o processo de financiamento, eles enviam vários aplicativos de crédito. Com esse fato em mente, instruí especificamente o gerente de vendas que eu só queria adquirir financiamento com XXXX. Eu instruí especificamente o gerente de vendas a evitar comprar meu crédito porque não queria inúmeras consultas no meu relatório de crédito. Estou no processo de adquirir um empréstimo comercial e fui instruído a evitar esse tipo de atividade. O gerente de vendas não comunicou adequadamente minha autorização para enviar apenas 1 solicitação de crédito ao gerente financeiro, o que resultou na submissão não autorizada a vários credores. O gerente geral da concessionária admitiu falhas e pediu desculpas pelas submissões não autorizadas. Embora isso me traga consolo para saber que suas ações não foram maliciosas, as submissões acidentais prejudicaram meu relatório de crédito e capacidade de adquirir um empréstimo comercial. Eu desafiei as consultas com as agências de crédito e fui encaminhado para desafiá -las diretamente com os credores.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  Isso exigirá tempo e esfor\n",
            "======================================================\n",
            "Texto: Estávamos no processo de modificar nosso empréstimo à habitação quando o XXXX XXX encerrou a conta. Todos os documentos foram submetidos a ela. Lidamos com xxxx xxxx, xxxx xxxx, xxxx xxxx, xxxx xxxx, xxxx xxxx, xxxx e xxxx antes da execução duma hipoteca implorando por um adiamento. Todos esses são funcionários de perseguir o escritório da XXXX XXXX, sem assistência; Chase foi encerrada de qualquer maneira.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "\n",
            "\n",
            "Agora, intuitivemente, a\n",
            "======================================================\n",
            "Texto: Chase xxxx xxxx xxxx xxxx xxxx xxxx xxxx Eu faço uma compra {$ 1000.00} de pontos xxxxxxxx e nunca recebi meu desconto de 20 % (USD {$ 200,00}). Espero cerca de 3 meses, ligo para o atendimento ao cliente muitas vezes e nunca recebi meu benefício.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Estou muito frustrado e qu\n",
            "======================================================\n",
            "Texto: Enviei várias cartas certificadas pedindo validação dessa dívida que eles dizem que devo. Prova de um contrato assinado para esta dívida. Em vez de enviar meu pedido, recebi uma convocação ao Tribunal de Pequenas Reivindicações.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "\n",
            "Caso você tenha fornec\n",
            "======================================================\n",
            "Texto: Chase e XXXX xxxx fecharam minhas contas com dinheiro nelas. Eu estou empregado por si só aqui, algumas pessoas cheques levam mais tempo para limpar e me puniram por isso\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: .\n",
            "\n",
            "Lamentamos ouvir\n",
            "======================================================\n",
            "Texto: Item pela primeira vez em XXXX. Item listado para {$ 800.00}. Interesse imediato do texto de xxxx. O comprador disse que enviaria cheques de caixa para {$ 1900.00}. {$ 800,00} para o item. Uma dica {$ 50,00} para mim por retirar a listagem e segurar o item para ele. E {$ 1100.00} que eu sou para fornecer os motores quando eles pegarem o item de móveis. Aparentemente, ele está se mudando para a cidade, mas sua companhia comovente iria pegar o grande item de móveis para ele. Eu recebo caixas cheque e deposite -o. Isso limpa. Defino o horário de coleta e o comprador diz que o {$ 1100,00} deve ser enviado via xxxx xxxx. Eu disse não. Apenas dinheiro ou cheque pessoalmente quando eles o buscarem. Ele disse que com a Covid. Ele disse que eu já tenho seu dinheiro e isso é legítimo. Então enviei {$ 1100.00} para xxxx e tive um mau pressentimento sobre isso. Eu também pensei que tinha o dinheiro na minha conta. Ele me evita nos próximos dois dias. Então diz que sua esposa sofreu um acidente. Eles não estão se movendo e ele não precisa do item. O cheque de caixa é falso e enviei {$ 1100.00} que não posso voltar agora. Então, a fraude foi cometida contra mim. Entrei em contato para perseguir e, porque paguei mais de xxxx, eles não podiam me devolver meu dinheiro.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  O fed também não respondeu\n",
            "======================================================\n",
            "Texto: Olá, em xx/xx/17, fui ao meu banco Morgan Chase para retirar dinheiro, depois de inserir meu cartão de débito e entrar no meu alfinete que me pediram para fornecer identificação, bem, entreguei ao caixa, logo depois que ela ligou O gerente dela e eles começaram a me fazer muitas perguntas como o meu SSN onde eu moro etc ... eu pedi para recuperar minha identidade e eles se recusaram dizendo que eu pareço suspeito para eles, bem, comecei a ficar nervoso porque estavam falando muito alto que todos Clientes olhando para mim como um golpista ... depois de mais de 10 anos, fui tratado como um ladrão ... nunca experimentei esse tipo de comportamento em toda a minha vida e esperando que seu envolvimento o faça o último para este banco. Obrigado pela sua colaboração. Sinceramente. Xxxx\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: O JP Morgan Chase continua a se defender dizendo que eles estavam na direita e que a execução duma hipoteca em nossa casa era válida e basicamente \"pelo livro\" \", embora possa ter sido o fim do fim, não foi porque deveria O nunca chegou ao ponto de execução duma hipoteca em primeiro lugar. Eles me acusam continuamente de mentir dizendo que eu não recebi os documentos solicitados em tempo hábil quando o fiz muito (assim como o escritório de advocacia que era me ajudando na época) e eles também me acusam de mentir quando relato que um representante do JP Morgan Chase me disse que ela não podia fazer um pagamento temporário que eu fui criado para fazer e depois de fazer pagamentos temporários consistentes para vários meses. Ela disse que não podia aceitá -lo porque o empréstimo estava sob uma revisão de modificação. Bem, ... é por isso que os pagamentos temporários foram criados em primeiro lugar. Toda vez que relato o JP Morgan Chase, tudo o que eles fazem é enviar a mesma resposta, alterando apenas as datas na carta. Ninguém leva tempo ou se importa para investigar verdadeiramente meus relatórios, muito menos oferece uma resolução razoável ou satisfatória. Eles nos ferraram e me deixaram sem -teto por não processando nosso pedido de modificação de empréstimo imediatamente e depois virou -se e tenta culpar -o por não conseguir os documentos solicitados para eles em tempo hábil quando eu certamente o fiz. Relatei isso várias vezes e não paro de relatá -lo até que uma resolução satisfatória nos seja feita para esse desastre e xxxx no final deles.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Xxxx xxxx Precisa de todos os itens removidos que alguém está usando minhas contas de verificação de ID Chase fechadas\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "\n",
            "Todos os itens usados\n",
            "======================================================\n",
            "Texto: Recebi uma oferta de refinanciamento da Chase (minha empresa hipotecária atual). Ele afirmou que eles reduziriam meu interesse de 4,75 para 3,75 e minha APR seria 4,22. Liguei para Chase e trabalhei com xxxx xxxx. Ela foi muito legal e assegurando e, quando eu forneci o número da oferta na carta para ela, ele confirmou as taxas e me perguntou se eu queria um refinanciamento de 15 anos ou 30 anos. Eu disse a ela 30, porque eu gosto de um pagamento mensal mais baixo. Ela puxou meu crédito e disse que minhas pontuações foram ótimas; Xxxx 's. Preenchi todo tipo de papel e enviei tudo o que ela exigia de mim. Várias semanas depois, recebo uma ligação dela dizendo que o subscritor está exigindo que ela entre em contato comigo e faça uma contra -oferta de um refinanciamento de 20 anos. Eu gosto .... Tenho 21,5 anos para fazer um empréstimo de 30 anos. Eu quero um empréstimo mais longo, não mais curto, para que eu possa ter um pagamento mais baixo. Ela pediu desculpas e me disse que foi a primeira vez que ela foi obrigada a fazer isso e que ela deixaria o subscritor saber que eu queria o refi de 30 anos. Recebo uma ligação por dia mais tarde do XXXX, dizendo que não fui aprovado há 30 anos, eles só poderiam fazer o 20 anos. Eu disse a ela que a carta não me ofereceu um empréstimo de 20 anos. Ofereceu 30 anos. Eu nunca pedi um 20 anos. Ela não parecia saber por que eu estava sendo negada e disse bem, talvez seja porque sua APR será de 4,55 e, como não está caindo pelo menos meio ponto, o subscritor decidiu não aprovar. Eu disse de onde veio isso 4,55? Você me disse 3,75 juros e 4,22 de abril. Seria aprovado se você derrubar a APR que me disse que seria? Ela realmente não parecia saber como me responder. Eu disse a ela que sinto que estava enganado. Como se tivessem tentado puxar uma isca e ligar para mim e enviei uma mensagem para a página XXXX, aconselhando -os sobre o que havia sido colocado. Fui contactado por um bom nome de senhora xxxx, que parecia saber menos que xxxx, e ela também não foi capaz de me ajudar, nem me dar nenhuma explicação sobre o que aconteceu com minha oferta de empréstimo original. Tudo o que eu queria era refinanciar meu empréstimo e diminuir meu pagamento mensal. Tudo o que consegui foi uma grande perda de tempo e uma pontuação de crédito mais baixa.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Em xx/xx/2019, fui vítima de um golpe de computador. Minha placa XXXX foi carregada {$ 470,00} por xxxx xxxx. Eu nunca comprei ou recebi nada por xxxx xxxx. Fui forçado a essa situação por xxxx xxxx em xxxx, que se passou fraudulentamente como um serviço ao cliente xxxx e que assumiu o controle remoto do meu computador. Mais tarde, descobri que esse número não era xxxx e xxxx nunca funcionou para a empresa. Liguei para XXXX imediatamente após a acusação ter sido feita para detê -la e expliquei o que havia acontecido. Eu tinha certeza de que o {$ 470,00} seria colocado em status de disputa e o assunto seria resolvido. Eu deveria receber um formulário de disputa para mim- nunca chegou. Liguei para XXXX novamente e outro formulário de disputa deveria ser enviado para mim- nunca veio. Liguei de novo e descobri que nenhuma forma foi enviada para mim, mas que o assunto foi resolvido porque era fraude. Em XX/XX/2019, recebi um e -mail do XXXX dizendo que a disputa estava fechada e as cobranças eram válidas. Liguei para XXXX e fui instruído a enviar uma carta ao departamento de disputas. Se eu discordasse para reabrir a disputa- o que fiz. O XXXX lidou com essa queixa muito mal e ineficaz, nunca conseguindo meu lado da história por escrito e mentindo para mim sobre os formulários de disputa e sobre o assunto sendo resolvido, porque era fraude. Já era ruim o suficiente que um golpe tivesse sido perpetrado, mas pior que o XXXX não fez nada a respeito e ficou do lado do golpista.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Estou decepcionado com o servi\n",
            "======================================================\n",
            "Texto: Há 2 meses, recebi um cartão de crédito XXXX Chase para o qual não solicitei. Liguei imediatamente para o Atendimento ao Cliente da Chase para relatar minha preocupação/conta fechada. O atendimento ao cliente da Chase me enviou para o departamento de fraude. Eles me fizeram uma série de perguntas, colocaram um alerta na minha conta. Eu tenho XXXX CRITE CARD CARTO XXXX CARRO DE VISA DE CHASE ATRAVÉS DO PROGRAMA DE RECOLHAS DE XXXX) e me enviei algum material de educação de fraude gratuita. Eu verifiquei meu relatório de crédito, nada apareceu lá a partir de xx/xx/xxxx. Eu não sou membro da Amazon, minha esposa é.   Acabei de receber um telefonema da Chase hoje (xx/xx/xxxx) me perguntando se me inscrevi em um cartão da Amazon e disse que não. Ele começou a fazer perguntas (informações pessoais), então eu desliguei. Nada está no meu relatório de crédito, etc ... o que posso fazer ... quem devo ligar. Obrigado\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: .\n",
            "\n",
            "Eu recomendo procur\n",
            "======================================================\n",
            "Texto: 0n xx/xx/2021 Localizado em xxxx xxxx xxxx xxxx xxxx il persei atm em xxxx tentou depositar {$ 1300.00} em dinheiro no atm, atm, coma meu dinheiro não foi o depósito e nenhum dinheiro pode voltar fora do atm, Percebi o Chase Bank XXXX e eles me deram o número do caso, ainda aguardando o resultado da investigação.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "\n",
            "Gostaria de relatar este cas\n",
            "======================================================\n",
            "Texto: O JPMorGanhase nos disse em xx/xx/xxxx que tínhamos uma escassez de garantia de {$ 360,00} e poderíamos pagar por xxxx/xxxx/16 para manter nosso pagamento de hipoteca da mesma forma. Quando chegou o XXXX XXXX, havíamos mudado nossa apólice de seguro para proprietários para locatários e isso deveria ter produzido um pagamento de hipoteca mais baixo. Liguei para a primeira coisa em xx/xx/xxxx para descobrir que não tinha. O caos começou no xxxx xxxx com a 1ª pessoa com quem falei (para o qual tenho nomes/datas/vezes) dizendo que ela se esforçaria no pagamento xx/xx/xxxx, para que eu não fosse uma taxa de atraso enquanto ela conduziu uma investigação, incluindo outra análise de garantia. Ela também disse que, de volta ao XX/XX/XXXX, eles pagaram seguro de proprietários e, em seguida, os locatários no mês passado, para que pagaram duas vezes e esse foi o motivo do alto pagamento/escassez. Isso acabou sendo falso. Eles foram reembolsados ​​pelo pagamento do seguro de proprietários em xx/xx/xxxx. Ela também mudou de idéia na mesma conversa, afirmando que poderíamos pagar o valor antigo nesse meio tempo, e não o \"espera\". por xxxx/xxxx/16 para obter um pagamento mais baixo. Este próximo par de representantes disse que isso ocorreu porque mais tempo passou entre a 1ª análise de custódia e o 2º deixando a escassez sem remuneração. O último representante disse que eu poderia pagar o último valor do pagamento da hipoteca para cima a 2 meses. Eu fiz e adicionei {$ 150,00} para a escassez de custódia. Na terça Pague pela escassez e o dinheiro que pagamos para a escassez foi aplicado ao diretor! Eu tive que ligar ontem para consertar isso e eles disseram que estava sendo consertado \"enquanto falamos\". Cada representante se desculpando pelo \"erro\" do último e dizendo que eles \"não deveriam ter dito isso, eu não sei por que eles fizeram\". Parece-nos que eles são altamente subestimados ou inventando à medida que avançam! Atualmente, o agora {$ 98,00} (retirado do {$ 150,00} para compensar a diferença para pagar o maior pagamento de hipoteca) não foi aplicado à escassez de garantia. Por favor, ajude o CFPB, para quem recorremos quando uma grande empresa corporativa como a JPMorGricaChase assume nosso dinheiro suado como esse? Sou um professor mal pago e meu marido um trabalhador de colarinho azul empregado por um município da cidade.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Nós não temos recursos\n",
            "======================================================\n",
            "Texto: Eu fiz um depósito móvel de {$ 3500.00} na minha conta bancária Chase em xx/xx/19. Recebi uma mensagem informando que uma retenção de 7 dias úteis seria colocada no item devido a informações indicando que o item não pode ser pago.   Eu sei que isso é uma violação do Reg CC. Para colocar uma exceção em um item, o banco deve ter motivos específicos para duvidar da cobrança. Eles não entraram em contato com meu banco para verificar os fundos. Se tivessem, teriam sido informados de que os fundos estavam disponíveis. Não tenho histórico negativo com Chase (depósitos devolvidos, NSF, etc.), o cheque foi pago apenas para mim e endossei o cheque corretamente.   Não havia motivo legítimo para duvidar da cobrança neste item. O porão foi colocado no meu depósito arbitrariamente, que é uma violação do Reg CC.   Entrei em contato com Chase ontem para solicitar que a retenção fosse removida. O representante do atendimento ao cliente com quem falei me disse que a espera não poderia ser removida até que o item publicado, então eu teria que ligar de volta no dia seguinte. Mas ela me disse que colocaria uma nota na conta de que os fundos deveriam ser lançados com base em \"My File\". Liguei de volta no dia seguinte e recebi uma RSE diferente. Esse indivíduo reconheceu que a nota na conta afirmando que os fundos poderiam ser libertada, no entanto, ela se recusou a fazê -lo. Ela disse que precisava ver se os fundos haviam esclarecido primeiro. Ela me colocou em espera, voltou e disse que não havia esclarecido. Ela disse que entraria em contato com meu banco para verificar a colecionabilidade ( provando que isso não havia sido feito antes da retenção ser colocada). O banco verificou os fundos. Finalmente, ela concordou em liberar o porão.   Os fundos já estão disponíveis para mim. No entanto, o fato de essa retenção ter sido colocada no meu depósito basicamente automaticamente, sem a devida diligência da parte da instituição financeira, acredito que essa é uma questão sistêmica e que outros clientes estão sendo aproveitados sem perceber que o banco está violando regulamento. Acredito que essa instituição precisa ser investigada por violações habituais.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:    \n",
            "Como tal, estou entrando\n",
            "======================================================\n",
            "Texto: Cartão de crédito da Chase, os pagamentos de automóveis pararam em torno de xx/xx/xxxx, não estavam cientes de que eu estava no meio de uma mudança militar e me preparando para um XXXX.   Foi notificado pelo XXXX que eles haviam assumido minha conta do cartão Chase, então trabalhou com eles para resolver o problema em vez de resolver o problema.   Devido ao xxxx, eu não estava monitorando como deveria. (Agora eu verifico semanalmente e estou corrigindo meu creidt) pago ao cobrador de dívidas: xxxx xxxx (tenha carta de declaração dizendo que a conta paga integralmente de xxxx.   Pagamentos em 3 depósitos de xx/xx/xxxx a xx/xx/xxxx (a cada 15 dias) totalizando {$ 1900.00}.   Tentei contestar todas as três agências, sem ajuda. Apenas ajustado verborrage no relatório.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:    Esses pagamentos foram feit\n",
            "======================================================\n",
            "Texto: Chase está preso à contabilidade errada e parou de atender nossa hipoteca doméstica. O Chase nos enviou de volta nossos pagamentos xx/xx/xxxx e xx/xx/xxxx e está nos pedindo para pagar {$ 22000.00} para trazê -lo à corrente. Chase está errado. A partir de xx/xx/xxxx, nosso pagamento de hipoteca estava atual. O saldo restante foi {$ 350000.00} no livro de Chase (consulte a declaração; livro xxxx) como foi {$ 350000.00} em nosso livro (consulte o livro xxxx). Chase está errado ao dizer que não fizemos os pagamentos desde xx/xx/xxxx. Chase está fazendo um assalto e não atendendo a um empréstimo à habitação. A Lei Criminal de Occ/Cfpb poderia parar de Chase?\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "\n",
            "Sim, dependendo de como a\n",
            "======================================================\n",
            "Texto: Olá, reservei um voo em xx/xx/2020 para uma viagem usando pontos de recompensa XXXX Chase Ultimate e {$ 11,00} em dinheiro para uma viagem a xxxx em xxxx xxxx que saiu xxxx xxxx, 2020. em xxxxxxxxxx, 2020. 3 pernas do vôo. Entrei imediatamente em contato com o Chase para obter um reembolso total desde que a companhia aérea cancelou meu voo. Depois de mais de 3 meses de Chase, está me negando um reembolso dos meus pontos de recompensa e dinheiro de Ultimate, em vez de me oferecer crédito de viagem com a companhia aérea (que eu nunca solicitei e não quero). Como a companhia aérea cancelou os voos, não eu, e porque está saindo dos EUA, o Departamento de Transporte afirma que tenho direito a um reembolso total no meu formulário de pagamento original. Eu expliquei isso para perseguir e eles se recusam, dizendo que a companhia aérea não quer me dar meu reembolso. Também arquivei uma reclamação no Departamento de Transporte, no entanto, espero que você possa ajudar a resolver esse problema diretamente para mim. Paguei por um serviço em dinheiro e pontos e esse serviço nunca foi prestado por causa da companhia aérea. Anexei a documentação de tudo isso. Obrigado pela ajuda!\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, revisei meu relatório de crédito do consumidor xxxx e descobri o cartão Chase solicitou meu relatório do consumidor sobre xx/xx/xxxx sem meu conhecimento ou consentimento.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  Entrei em contato com a equipe\n",
            "======================================================\n",
            "Texto: Em xxxx/xxxx/2016, meu pagamento xxxx foi depositado na minha conta líquida (encerrando o inxxxx) no valor de {$ 2000.00}, pude transferir os fundos para minha conta corrente pessoal naquela manhã, essa conta foi negativa {$ 3,00} i Acredite (presumi que os fundos já foram pagos depois de fazer a transferência. Agora, na sexta -feira XXXX/XXXX/16 Manhã quando verifiquei minha conta, havia um {$ 2000,00} adicional (Acct terminando em xxxx). Pensei que outro depósito foi feito , porque pude transferir esses fundos com sucesso. Na época, todas essas transferências estavam acontecendo do meu telefone celular, não percebi que não tinha meu cartão de débito, agora estamos em {$ 4000.00} mais meu salário Do meu trabalho {$ 520,00} que foi depositado naquela sexta -feira xxxx/xxxx/16. Agora eu sou uma mãe solteira de xxxx que acabou de me afastar de uma situação xxxx, já estava em paz porque eu seria capaz de pagar todas as minhas contas no tempo da primeira vez na minha vida. Então, de repente, quando eu verifico minha conta em xxxx/xxxx/16 I 'M Negativo - {$ 320.0 0}, então fiquei furioso porque acabei de configurar todas as verificações de mim para passar por isso na semana seguinte. Liguei para o atendimento ao cliente, eles disseram que eu precisaria entrar em uma filial, entrei na filial e eles dizem que não podem fazer nada no nível da loja, a menos que eu estivesse fazendo um depósito para cobrir o negativo, para que ele n ' t acumula uma taxa. Liguei de volta para o atendimento ao cliente e, naquela época, fiquei irado porque não tinha meu cartão e sem fundos, eles me dizem que era uma \"falha\" no sistema, até começou a ler acusações que eu não fiz Mesmo fazendo, arquivei uma disputa que resultou em negação. Agora minha conta está exagerada {$ 860,00} e cerca de xxxx disso está em taxas bancárias da minha conta e-verifica. Desde então, fechei minha conta líquida e trabalhei no fechamento da minha conta comercial , com medo de outra falha em seu sistema.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Veja o anexo Esta é uma queixa para assistência. Persiga os pagamentos de postagem inadequados de ardósia. Foram fornecidas lnstruções para publicar o XXXX para a data de vencimento XXXX. Agora, aparece no extrato um pagamento mínimo devido ao XXXX, que está incorreto 'Slate Chase não conseguiu responder à solicitação de email e carta original.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "Primeiro, eu recomendar\n",
            "======================================================\n",
            "Texto: Meu marido faz cobranças periódicas em xxxx xxxx em nome da minha filha xxxxyo. Como o XXXX mantém automaticamente as informações do cartão de crédito arquivadas, ele precisa dar a etapa extra para remover nosso cartão de crédito após cada compra. Se pudéssemos obter um histórico de atividade no XXXX, você veria que ele faz isso toda vez que compra algo de xxxx xxxx em nome da minha filha. XXXX mantém essas informações em seus arquivos de dados. Em xx/xx/20, ele autorizou uma cobrança no xxxx xxxx por {$ 19,00}. Ele removeu o cartão antes de devolver o tablet a ela.   Sem o conhecimento para ele, xxxx manteve o cartão em arquivo e minha filha cobrou acidentalmente {$ 1500,00} em xxxx cargas entre xx/xx/20 - xx/xx/20. Aprendi sobre as cobranças em xx/xx/20 e entrei em contato com xxxx para obter um reembolso. Eles disseram que não se qualificou, já que nosso cartão estava em seu sistema. Meu marido tem muito cuidado ao remover o cartão após cada compra e diz que foi removido.   Para provar que ele o removeu, pedi a XXXX para documentação sobre a atividade da conta (cada clique e comando fornecido) para xx/xx/20 e me disseram que eles não poderiam fornecer isso para mim. Esta é a única maneira de provar que meu marido havia removido o cartão, para que não receba o recorrente de receber essas acusações não autorizadas.   Até o momento em que este artigo foi escrito, nossa empresa de cartão de crédito, o Chase Bank, também não nos reembolsará. Eles dizem que o XXXX contesta nossa alegação de que essas cobranças não foram autorizadas para que não creditam as cobranças. Além disso, os serviços de fraude do Chase Bank falharam nos notificando cobranças internacionais (XXXX está baseado em xxxx), como eu havia indicado em nossos alertas e o fato de que uma empresa estava cobrando nosso cartão repetidamente - xxxx cobranças em menos de alguns Horas - e não me enviou um alerta ou negar as cobranças com base em atividades suspeitas. Sinto -me como XXXX e Chase não estão oferecendo a proteção adequada contra as cobranças não autorizadas que eu incorri.   Desativamos todos os recursos de compra em xxxx devido a esse problema. Fizemos tudo o que sabíamos fazer para evitar cobranças não autorizadas e, no entanto, quando aconteceu, o XXXX não nos reembolsará ou fornece os dados necessários para mostrar que removemos o cartão. Minha queixa com Chase é que eles não me notificaram quando as acusações suspeitas estavam ocorrendo (não havia alerta que eu pudesse me inscrever quando as acusações são enviadas repetidamente em um curto período de tempo) e quando as notifiquei em xx/xx/ 20 Que as cobranças não foram autorizadas, elas não negam o pagamento a XXXX.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Fiz uma compra para ingressos para xxxx xxxx on -line com xxxx xxxx para r xxxx de volta em xx/xx/xxxx no meu cartão de débito Chase. Chase fechou minha conta em xx/xx/xxxx. Xxxx xxxx me notificou em xx/xx/xxxx por e -mail que o show foi cancelado e um reembolso foi emitido de xxxx no cartão que ele carregou. Liguei para o Chase Bank sobre isso em xx/xx/xxxx e xx/xx/xxxx e eles agem como se não tivessem recebido esses fundos que tenho provas de xxxx xxxx da transação de retorno, então por que não me acertar Retorne o reembolso a assentos vívidos. O Chase Bank é uma fraude que eles mantiveram em 5.000 do meu M no EY por três semanas quando cancelaram minha conta em xx/xx/xxxx falando sobre atividades suspeitas.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Enviei uma reclamação\n",
            "======================================================\n",
            "Texto: Etapa 1: O que é essa reclamação? Os Serviços de Membro do Cartão da Chase continuam a reforçar que eu fiz uma cobrança válida, mas a empresa não me diz o que foi compra para o dinheiro renderizado e quem realmente é a empresa porque não reconheço a cobrança. Além disso, apesar de várias tentativas de resolução e mensagens de email em seu sistema de mensagens seguras, elas não parecem ler as solicitações de email e responder a perguntas que nunca foram feitas. A solicitação é fornecer o nome da empresa (seu nome real e não um negócio de fazer como nome falso que eu não reconheço e também para me dizer o que foi comprado para o {$ 150,00} na data xx/xx/2019.   Etapa 2: Que tipo de problema você está tendo? A empresa não está fornecendo respostas claras para as perguntas feitas e não está resolvendo a disputa. A empresa continua dizendo que a cobrança é válida, mas não produzirá o nome real da empresa que recebeu o dinheiro, as informações de contato da empresa e o produto ou serviço para o qual a cobrança foi feita. A Companhia (Chase) tem um contrato com o fornecedor, caso contrário, o fornecedor não aceitaria o cartão de crédito da Chase. A Chase tem um dever e a obrigação de me fornecer a empresa nomear suas informações de contato para incluir número de telefone, endereço, site, endereço de e -mail e produto ou serviço que foi comprado. Chase não está fornecendo essas informações, apesar de várias tentativas.   Etapa 3: O que aconteceu? Aparentemente, uma cobrança foi feita por {$ 150,00} em xx/xx/2019 para uma empresa chamada xxxx fl xxxx xxxx. Eu não reconheço a empresa. O nome não faz absolutamente nenhum sentido para mim e não conheço o produto ou serviço que foi comprado. Eu contestei a cobrança várias vezes apenas para perseguir me dizer que a cobrança é válida. No entanto, o Chase não produzirá o nome real da empresa, suas informações de contato, nem o produto ou serviço adquirido. Não há motivo válido para ocultar e não produzir essas informações para mim (o comprador) do produto ou serviço.   Etapa 4: De que empresa é essa reclamação? CHASE CARD MEMBROM SERVIÇOS.   Etapa 5: Quem são as pessoas envolvidas? Os Serviços de Membro do Card Card e uma empresa chamada XXXX FL XXXX XXXX.   Etapa 6: Resolução solicitada: Dado que o Chase não produzirá as informações solicitadas e não pode ser determinado a quem a cobrança foi feita, nem que produto ou serviço foi compra, solicito que a compra seja invalidada e um reembolso ao meu cartão seja feito de {$ 150,00} para a compra xx/xx/2019.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Em XX/XX/2017, fui informado de uma conta aberta fraudulentamente em meu nome no Chase Bank. Entrei em contato com o escritório do meu xerife local, todas as agências de crédito, meus bancos e minhas corretoras. Desde então, descobri várias contas criadas em meu nome e me comuniquei com o Chase Bank várias vezes durante esse período. Chase me notificou que certas contas foram fechadas, que entraram em contato com as agências de crédito e que não sou responsável. No entanto, ainda recebo cartas, e -mails e contas deles com diferentes números de conta. Eu enviei uma queixa ao escritório do procurador -geral da Flórida.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  Finalmente, após dois an\n",
            "======================================================\n",
            "Texto: Eu tenho inúmeras disputas neste cartão, levei -o até os escritórios executivos e não tive ajuda. Devido ao número de disputas em que fui cobrado duplo por estadias de hotel. Uma vez nos sites de reservas on -line e uma vez no hotel, onde paguei minha conta.   Agora estou tentando conciliar isso, mas é muito difícil, pois tenho cerca de 4 números de contas diferentes devido ao cancelamento de meus cartões várias vezes. Minha pontuação de crédito caiu de xxxx para menos de 600. Passei horas tentando conciliar essas cobranças e continuarei até que isso seja corrigido.   Como resultado de suas investigações anteriores, não me levou a lugar algum e eles estão dizendo que devo a eles o dinheiro. Vou conciliar esta conta e depois pagar apenas o que devo. Se Chase continuar, eles podem me levar a tribunal. Eu já tive o suficiente !!!\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "No momento, recomendo que\n",
            "======================================================\n",
            "Texto: Fui aprovado para o empréstimo SBA EIDL, de acordo com os requisitos de qualificação e qualificação da SBAS que fui aprovado como um xxxx xxxx xxxx. Passei por um processo de subscrição completo e completo e a SBA me aprovou para o empréstimo, afirmei que, porque sou xxxx xxxx e não tenho uma conta comercial, eles afirmaram que está tudo bem. E eles também desembolsam fundos em contas de verificação pessoais para pessoas que são xxxx xxxx e não têm problemas. Por outro lado, persegue minha conta e exige que eu tenha um número EIN, quando legalmente não preciso ter um número EIN, e tudo o que faço é arquivar um cronograma C. Eles não têm o direito de solicitar essa documentação minha Quando estou trabalhando como um indivíduo xxxx xxxx e eles não são os que estão me emitindo o empréstimo. A SBA está financiando e determina a qualificação e a elegibilidade. Conversei com inúmeros advogados, e todos declararam que Chase deve rejeitar os fundos de volta à SBA se não estiverem aceitando em uma conta corrente pessoal ou deve liberar os fundos de volta para mim. Eles não têm o direito de manter meu dinheiro refém.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta:  Espero que eles entendam que tad\n",
            "======================================================\n",
            "Texto: Sou cliente há 4 anos com Chase, em xx/xx/xxxx, eles me cobram uma taxa de processamento legal de {$ 75,00}, que tornou o saldo da minha conta negativo e, em seguida, continuaram me cobrando taxas porque minha conta foi negativa. Quando liguei para Chase, eles me disseram que teriam que me transferir para o especialista jurídico deles e fiquei em espera por longos períodos de tempo até que acabassem desligando. Eles também não estão me permitindo fechar a conta também.   Xxxx xxxx - foi o gerente que tomou a decisão.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "Lamentamos ouvir so\n",
            "======================================================\n",
            "Texto: Em xx/xx/xxxx, fui divorciado da minha ex-esposa. Minha ex-esposa e eu tínhamos uma renda combinada de cerca de US $ 125 mil. Após o divórcio, fiquei com a casa e tive que remover o nome da minha esposa da hipoteca para mantê -la. O divórcio me deixou assumindo todas as dívidas que já foram compartilhadas com minha ex-esposa e prejudicaram meu valor de crédito. Eu me inscrevi para perseguir a hipoteca para uma modificação. Depois de fornecer minhas informações financeiras em xx/xx/xxxx, o Chase Mortgage concordou com uma modificação apenas diminuindo minha hipoteca em {$ 20,00}. Embora minha renda familiar tenha diminuído 50 %, tentei continuar pagando o empréstimo recém -reduzido por meio de xx/xx/xxxx. Eu deveria vender a casa entre xx/xx/xxxx-xx/xx/xxxx, mas fui demitido em xx/xx/xxxx, então não coloquei a casa no mercado até o inverno de xx/xx/ Xxxx. Não pude solicitar outra modificação por um ano, então fiz o meu melhor para continuar a manter minha casa. Sem precisão, acabei arquivando a falência do capítulo xxxx antes da execução da execução horária programada em xx/xx/xxxx e entregando minha propriedade. No início do XX/XX/XXXX, recebi um cheque de ação coletiva de ação de ação de que Chase havia sido processada por discriminação ao conceder empréstimos ruins a minorias. Destruí o cheque porque senti que também havia sido vitimado, e que Chase havia me dado a modificação mal feita para que eu acabasse perdendo minha casa em um ótimo bairro com um excelente mercado. Agora, estamos passando por uma venda a descoberto por uma quantia que cobrirá o empréstimo por cerca de US $ 170 mil. Chase estava certo ao projetar que eu perderia minha casa com a diminuição da modificação {$ 20,00} e que a casa venderia rapidamente no mercado local.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Meu nome é xxxx xxxx e minha reclamação é com a Chase Mortgage Company. Dois anos atrás, recebemos um mod de empréstimo da FHA deste co. De acordo com o nosso empréstimo, nosso empréstimo é considerado em boa posição, a menos que fique 90 dias para trás. Atualmente, estamos dois meses atrasados ​​devido a dificuldades financeiras. Tentamos fazer um pagamento para nos impedir de deixar 90 dias para trás e eles estão recusando o pagamento. Eles querem nossa casa. Estamos planejando vender a casa nos próximos 6 meses, pois temos muito patrimônio. Por favor, ajude eles estão tentando levar nossa casa.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "\n",
            "Agradeço pela sua\n",
            "======================================================\n",
            "Texto: Eu tive essa hipoteca desde xx/xx/xxxx. Minhas taxas foram tão altas quanto 14 % durante esse período. Tive lutas com os pagamentos como resultado da economia e a queda nas vendas de imóveis nos últimos anos. A taxa predatória que ainda existe na minha hipoteca não ajudou a ser importante. Minha taxa ainda é XXXX ou XXXX Points acima da taxa de mercado agora após as modificações xxxx. Devo quase tanto quanto emprestei xxxx anos atrás. Isso é mesmo depois que eles afirmam que cobraram {$ 120000.00}, que também é uma ação injusta, já que eu tenho que registrar isso nos meus impostos e pagar quase {$ 30000,00} em impostos que não posso pagar. É óbvio que o Chase não quer tornar minha casa acessível para mim, embora eu pague taxas predatórias extremas há anos. Além disso, meu empréstimo é ilegal, pois foi originado de um credor não licenciado. (Xxxx xxxx xxxx), agora fora do negócio e sem licença de empréstimo da NC. Foi vendido para xxxx xxxx, xxxx xxxx xxxx, xxxx xxxx, hipoteca EMC e agora perseguir. Precedente foi estabelecido na NC que empréstimos hipotecários originários de empresas não licenciadas ou mesmo indivíduos estão sujeitos a cancelamento pelo estado da NC. Eu tentei fazer a coisa certa em não entrar com um processo nesses motivos. É duvidoso que esse empréstimo passe no teste de cheiro de securitização. Eu trabalhei junto com Chase para modificar e fazer arramgas que não foram do meu interesse financeiramente e me mantém em um vinculativo. Meus pagamentos foram tão altos quanto {$ 4500,00} por mês. (quando teve taxa variável). O acordo de pagamento mais recente que me foi oferecido nesta semana é um excelente exemplo. Eu já sou desafiado financeiramente e Chase quer um pagamento de redução de $ xxxx e Ober XXXX por mês. Fiz extensos reparos na minha casa e quero mantê -la. Ele ainda tem base de base que xxxx xxxx estimado em xx/xx/xxxx para mais de {$ 30000.00}. Desde então, piorou e muitas das minhas portas não fecharão adequadamente e a lareira está se separando da parede. Isso afeta o valor e a segurança da minha casa. O valor que foi colocado em minha propriedade durante esse processo de modificação é inflado porque ninguém esteve dentro ou no local para ver os problemas. Fui desqualificado injustamente por uma modificação decente devido a informações falsas sobre a avaliação. As casas da minha área são mais velhas, não com eficiência energética e permanecem no mercado em média de XXXX anos ou mais. Os comps usados ​​no valor estão fora da minha zona de comp. Isso pode ser considerado fraudulento. Eu realmente preciso de uma redução no meu saldo para corresponder ao valor real e uma taxa concomitante ao mercado ou abaixo, pois existem programas do presidente para ajudar os proprietários. Caso contrário, esses empréstimos precisam ser litigados no tribunal para provar que é ilegal e que paguei demais todos esses anos.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta:  Preciso de uma forma de impor\n",
            "======================================================\n",
            "Texto: Verificamos nosso saldo em xx/xx/xxxx e ele tinha {$ 240,00} na conta, fizemos algumas transações de alimentos que foram deduzidas. Verificamos o saldo no mesmo dia e fizemos um depósito por {$ 300,00} em xx/xx/xxxx. Anexamos nosso deslizamento de depósito. Após esse depósito, fizemos um pagamento pelo pagamento do carro.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  Novamente, anexamos desl\n",
            "======================================================\n",
            "Texto: Todo o meu AutoPay do meu cartão de crédito foi cancelado sem notificação prévia enquanto eu ainda esperava do agente no banco para resolver meu caso.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Eu entendo que o cancelam\n",
            "======================================================\n",
            "Texto: Abri um 2 verificando uma conta de poupança no Chase Bank. Eu gerenciei as contas corretamente nunca tendo um problema no banco. Percebi que havia alguma atividade não autorizada nas minhas contas e entrei em contato com Chase. Eles abriram uma disputa para mim e sugeriram que eu abrisse uma nova conta e fechei as contas existentes. Eu estava esperando meu depósito direto para postar e depois fechei as contas. Eu vou fazer login e fiquei restrito apenas para saber mais tarde que perseguir fechou todas as minhas contas. Recebi uma carta pelo correio afirmando que não havia devolvido o trabalho em papel oportuno. O problema é que nenhuma papelada foi solicitada a mim. Sinto como se tivesse sido inadequado pelo Chase Bank. Sinto -me envergonhado, discriminado e violado pelas ações que Chase tomou contra mim.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Nossa empresa alugou recentemente 4 veículos e o arrendamento foi financiado pelo JP Morgan Chase IE. Chase Auto Finance. Isso foi organizado pela concessionária de carros. O Bank Fills nossa empresa e os pagamentos de arrendamento são feitos pelo gerente da nossa empresa em tempo hábil. O JP Morgan Chase, por algum motivo, não tem creditado os pagamentos adequadamente, ou seja. Eles são creditados mais a algumas contas e algumas outras são relatadas como delinqüentes.   O representante do banco me chamou em xxxx/xxxx/xxxx e foi explicado a ela que essa era uma conta da empresa. Eu disse a ela que ela deveria estar ligando para o escritório, pois o problema será abordado pelo gerente do escritório. Ela mencionou que havia ligado e ninguém respondeu. Quando questionada mais sobre a que horas ela ligou, como durante o horário comercial, não há razão para que ela não consiga alcançar alguém lá, ela não conseguiu me dar os detalhes. Foi mencionado que eu passarei isso ao gerente do escritório e essa questão será abordada por ela em alguns dias quando ela voltar de suas férias. O representante foi bastante rude ao telefone e ela afirmou que continuará me ligando.   Em alguns dias, quando o gerente do escritório estava de volta, ela ligou para eles e foi informada de que eles haviam cometido o erro e creditaram algumas contas com erro. Eles disseram que levará 20 dias para resolver o problema. O gerente do escritório mencionou que isso não era apropriado como o pagamento foi feito e o erro foi da parte deles, para que eles devam corrigi -lo em breve. O gerente do escritório me mencionou que o representante também era rude com ela.   Apesar de tudo ser abordado como observado acima, continuei recebendo ligações repetidas, ou seja. 6 on XXXX/XXXX/XXXX, 4 on XXXX/XXXX/XXXX, 4 on XXXX/XXXX/XXXX, 4 on XXXX/XXXX/XXXX, 5 on XXXX/XXXX/XXXX, 4 on XXXX/XXXX/XXXX. Nesse ponto, eu tive que bloquear suas ligações.   Então recebi uma carta datada de xxxx/xxxx/xxxx informando que eu havia ficado para trás no meu pagamento de {$ 480,00}. Conversei com o gerente do meu escritório e fui instruído a ignorá -lo como o assunto havia sido abordado. Hoje, hoje de manhã xxxx, acho que minha pontuação de crédito \"caiu cerca de 80 pontos\" porque o banco havia colocado 1 pagamento em atraso na minha conta.   É bem provável que esse comportamento arrogante dos bancos e sua equipe não esteja isolado. Sua tentativa de intimidar clientes individuais que não têm recursos para enfrentá -los não deve ser tolerada. Eu apreciaria se você pudesse investigar esse assunto e tomar as medidas apropriadas.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Reclamação: O hipoteca (Chase Bank) está cobrando mais do que o valor permitido para minha conta de custódia. A conta de custódia deve cobrar para pagamento do meu seguro residencial e impostos sobre a propriedade. Liguei para o Chase Bank, acredito em XX/XX/2018 para discutir seu erro, mas eles não remediaram a queixa. Detalhes: Meu seguro residencial anual combinado de {$ 860,00} e impostos anuais da propriedade {$ 2500,00} Total o valor de {$ 3400.00}. O Banco Chase está cobrando em garantia um valor total anual de {$ 4400,00} (ou seja. {$ 370,00} x 12 = {$ 4400.00}). Por HUD, o máximo que um credor pode coletar é o valor anual total devido ({$ 3400,00}) mais dois meses ({$ 3400,00} / 12 = {$ 280,00} x 2 = {$ 560,00}). Por regras do HUD, isso totaliza um valor máximo permitido de garantia de {$ 3900.00} (ou seja. {$ 3400.00} + {$ 560.00} = {$ 3900.00}). O Chase Bank está coletando {$ 460,00} sobre o máximo permitido (ou seja. {$ 4400.00} - {$ 3900.00} = {$ 460.00}.). O Chase Bank precisa corrigir o valor da coleta de garantia para cumprir as regras do HUD.\n",
            "Predição Tradicional: Outros\n",
            "Resposta: \n",
            "\n",
            "Solução: Queremos\n",
            "======================================================\n",
            "Texto: O patrimônio residencial foi modificado em xxxx ... Chase não atualizou o sistema e continua ligando dizendo que o empréstimo está em atraso .....\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "\n",
            "A partir de xxxx, alter\n",
            "======================================================\n",
            "Texto: Estamos passando pelo processo de obter uma hipoteca através de perseguir e sofremos de serviço incrivelmente ruim e o banco não conseguiu responder às nossas perguntas em tempo hábil. Nosso oficial de empréstimo original parece ter sido suspenso e seu chefe deve estar lidando com o nosso caso, mas ele claramente não tem tempo ou interesse em nosso empréstimo, por isso ficamos com o processador de empréstimo que está em grande desvantagem devido a O oficial de empréstimos está estragando as informações que foram enviadas repetidamente para a subscrição (incorretamente) e sua própria incapacidade de manter direto o que enviamos para ela. Compondo isso é a falta de conhecimento da papelada de que precisamos e das datas nas quais esta documentação expirará ou o compromisso do empréstimo expirará sem a papelada. Ninguém na Chase parece ser capaz de agir de maneira oportuna e profissional em responder a perguntas ou devolver chamadas (se elas forem devolvidas; o que não é frequentemente). Para receber uma ligação de volta, sempre precisamos recorrer à escalada, onde alguém fora da linha de relatório faz check -in sobre o que está acontecendo. Em seguida, recebemos chamadas de volta com as reivindicações obviamente falsas de que eles ligaram de volta várias vezes e optaram por não deixar uma mensagem. Eu verifiquei meus próprios registros telefônicos (no meu telefone) e entrei em contato com meu provedor de telefone e não há nenhuma ligação além das que atendemos ou onde eles deixaram uma mensagem. Em 2015, é um pouco insultuoso ser mentido em algo tão fácil de validar. Precisamos que alguém assuma a responsabilidade por nosso empréstimo na Chase e Chase parece incapaz de fazer isso por conta própria.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Não é possível vincular a conta ao aplicativo de poupança xxxx xxxx xxxx\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  através do Facebook. No entanto,\n",
            "======================================================\n",
            "Texto: Fiz uma reserva no xxxx xxxx xxxx xxxx em xxxx, tx para xx/xx/19 a xx/xx/19 para {$ 75,00} por noite. Quando fui cobrado, as cobranças eram xxxx, xxxx, xxxx, xxxx e xxxx por noite mais impostos. Notifiquei imediatamente a recepção e o funcionário disse que ela corrigiria as acusações. Mais tarde, quando não fui creditado, liguei e novamente. O funcionário da recepção novamente me disse que eu seria cobrado {$ 75,00} por noite mais imposto sobre vendas. Após o quarto telefonema, me disseram que o computador ajusta automaticamente as taxas com base na capacidade. Aconselhei que eles não poderiam ajustar minha taxa sem aviso prévio, porque eu tinha uma reserva. Aconselhei a empresa de cartão de crédito da Chase o problema e eles disseram que eu tinha que pagar o que foi cobrado. Expliquei que fiquei no Hotels Milless's of Times e nunca fui mais cobrado do que o valor acordado. Além disso, expliquei que toda vez que uma pessoa chega em um hotel, assina um pedaço de papel com a taxa. Chase disse que eles não precisam de assinaturas e eu teria que pagar o que quer que cobrar pelo meu cartão. Aconselhei que isso era uma violação da lei e não pagaria o valor sobrecarregado de {$ 110,00}. Além disso, aconselhei que queria encerrar minha conta e ser reembolsado a taxa anual que se deve ao XXXX. Como não posso fazer negócios com uma empresa fraudulenta, quero que a conta seja encerrada, um contrato que não pagarei o {$ 110,00} em excesso e um reembolso pela taxa anual. Não usarei o cartão devido às ações fraudulentas da Chase. Usarei o XXXX XXXX, pois eles não responsabilizam o titular do cartão por fraude.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Estou escrevendo para relatar um erro bancário que está me afetando de uma maneira enorme. Meu nome é xxxx xxxx. Meu ssn xxxx, minha data de nascimento é xx/xx/xxxx. Tentei resolver esse problema com o JP Morgan Chase, mas eles se recusaram a me dar uma solução justa. Eu tenho uma conta no JP Morgan Chase Bank. Informações da conta: JP Morgan Chase-Auto Finance. Data aberta: xx/xx/xxxx. Mais alto saldo {$ 50000.00}.   Processei meu pagamento pelo XXXX dentro do prazo (dois dias antes da data de vencimento), como sempre faço todos os meses. Desta vez, paguei através do banco on -line, procurando uma maneira de facilitar para mim. Por qualquer motivo, o site indicou que a transação foi bem -sucedida.   Nunca recebi uma inserção de atraso no pagamento ou pelo menos uma ligação da Chase para me avisar sobre meus relatórios de obter um sucesso, isso é uma violação clara da Lei de Relatórios de Crédito Justo. Eu nunca estive atrasado na minha vida. É a primeira vez que um pagamento atrasado é relatado no meu histórico de crédito e é por causa de um erro bancário. Nunca recebi uma inserção de atraso no pagamento por correio. Isso é negligência, prática comercial injusta, quebra de contrato e a lista de violações pode continuar assim que me sentei com meus advogados.   Devo dizer que estou tentando configurar o pagamento automático no site do JP Morgan Chase há mais de 1 mês, mas não consegui, porque este site não é amigável, é muito difícil de entender. É por isso que processei um pagamento em tempo dois dois dias antes da data de vencimento. Estive sempre disposto a processar meus pagamentos a tempo e é por isso que tenho uma excelente pontuação de crédito, mas essas informações erradas relatadas às agências de crédito estão me matando. Parece que o Chase Bank intencionalmente dificulta esse pagamento automático on -line e com certeza não é a primeira vez que você aqui.   Exigo uma carta de exclusão da empresa mencionada o mais rápido possível para que eu possa fornecer à minha empresa hipotecária e aprovar os interesses que desejo. Se o Chase Bank se recusar a fornecer carta de exclusão por meio do portal do CFPB, não pensarei duas vezes em ter esse problema e ser decidido por um juiz em um tribunal. Graças a quem lê minha queixa.   Consulte o extrato bancário em anexo como prova de fundos no momento em que o pagamento com o Chase foi vencido. Provar que esse não era um fator de dinheiro ou mesmo o pior que eu não gostaria de pagar.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:    Como pode ver, o fluxo\n",
            "======================================================\n",
            "Texto: O JPMorgan Chase, duplo cobra juros de itens pagos, cobrança dupla. Conforme explicado ao XXXX XXXX, advogado de Chase, XXXX, que não teve a cortesia de devolver uma ligação e investigar, mas, desagradável, tinha outros funcionários e nos assedia, enquanto se recusava a valores disputados de crédito. O Chase se envolve ativamente em juros com dívidas pagas no prazo (cobrança do ciclo duplo), embora a conta seja paga um valor substancialmente, os juros ainda são impostos ao saldo como nenhum pagamento ocorreu e, além de pagamento, além de um ramo não creditado imediatamente e Chase cobra juros diários à medida que o pagamento foi realizado no dia seguinte. Os funcionários da Chase são abusivos, tratam para relatar informações depreciativas ao Credit Bureau e até fizeram chamadas ao chegar a nossa propriedade para cumprir a dívida do cartão de crédito, embora este seja um empréstimo inseguro.   De acordo com a APR e os cálculos, Chase trapaceou fechado para {$ 300,00} até agora.   Perseguir postagens adicionais em nenhum lugar on -line ou os termos ou acordos necessários por lei, e eles tendem a ligar para o horário de ligação do passado.\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  A duração do ciclo dupl\n",
            "======================================================\n",
            "Texto: Observe que o problema que estamos tendo não é uma seleção no \"melhor descrito\" 'de cima.   Estou escrevendo no XXXX e somos a seguradora de título na propriedade em questão. Chase mantém um julgamento que precisamos limpar, porque não foi interrompido na ação de encerramento do assunto. Seu devedor é o proprietário anterior que foi encerrado. Nosso cliente atual, o autor e o vendedor da propriedade precisam transmitir um título claro na venda do REO, e Chase está proibindo que isso ocorra. Aqui estão a série de eventos para esta edição: XX/XX/XXXX: XXXX Conselho, xxxx enviou uma estipulação proposta ao departamento jurídico do JPMorgan Chase em xxxx de solicitando que executem para extinguir a garantia de acordo com a execução hipotecária. Chase não respondeu. XX/XX/XXXX: XXXX OFERTAMENTO FAXO DO FAXO DE ALTO PARA ALGUMA DEPARTAMENTO DE SEUS JULGAMENTO/AVISO DE FALÊS E DEPARTAMENTO DE CORRESPONDÊNCIA DE JULGENGEM. Tentamos resolver a garantia e obter uma liberação. XX/XX/XXXX: XXXX Solicitação XXXX por fax para os dois departamentos. Xx/xx/xxxx: também enviou um email para xxxx, xxxx, de Chase pedindo sua assistência, já que ninguém na Chase respondeu. Ele estava mais preocupado com a forma como obtivemos seu e -mail do que em nos ajudar. XX/XX/XXXX: XXXX FAXED XXXX SOITE para revisão de julgamento. Xx/xx/xxxx: solicita xxxx por email para o advogado xxxx, pois nunca tivemos uma resposta. Novamente, sua principal prioridade foi como obtivemos seu endereço de e -mail. Xx/xx/xxxx: solicita xxxx por email para o advogado xxxx solicitando que ele filtre nossa solicitação para a parte correta. Ele ligou e deveria fazê -lo, no entanto, não ouvimos falar de outras pessoas em Chase. Xx/xx/xxxx: xxxx xxxx estava tentando nos ajudar e alcançou o xxxx, com quem eles tinham contato em outro arquivo. O XXXX informou que seus consultores jurídicos não executariam uma estipulação porque a garantia em questão foi paga e eles enviaram o comunicado ao proprietário executado (ex -proprietário), xxxx em xx/xx/xxxx. XXXX XXXX pediu um original duplicado, mas eles não produziriam uma vez até que tivessem autorização do proprietário. XX/XX/XXXX: XXXX Pedido por fax para revisão de julgamento e departamento de liberação de garantia, com as informações que tínhamos que a garantia foi paga; pedindo uma duplicata e deu a eles a autorização do proprietário atual. Afirmamos que estaríamos apresentando uma reclamação no CFPB se não recebermos resposta deles. Chase nos chamou no dia seguinte, respondendo à nossa carta, afirmando que eles não nos dariam a duplicata porque queriam o consentimento da XXXX. Expliquei que isso não seria possível, pois ele era excluído involuntário. Chase não parecia se importar, apesar do fato de eu ter dito a eles que teríamos que prosseguir com nossa reivindicação.\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "\n",
            "Neste caso, a melhor\n",
            "======================================================\n",
            "Texto: Recebi um aviso de que meu empréstimo privado do JP Morgan Chase Student estava sendo transferido para xxxx (xxxx. No momento em que paguei minha conta antes de ser transferida. Minha primeira conta da AES tinha um saldo xxxx. Minha segunda conta, me deixou denominar a mente E paguei 3 dias atrasado (paguei mais do dobro do que devia para o mês). Meu pagamento foi devido no xxxx e paguei no XXXX.   Ontem, xxxx/xxxx/15, recebi uma carta de coleta do xxxx xxxx, no valor de {$ 15,00}. XXXX REF # xxxxcreditor: JP Morgan Chase Bank, n.a. Balanço de empréstimo: $ xxxxpast devido: $ xxxxi chamado Chase para verificar por que recebi uma carta de cobrança e eles disseram que teriam que me conectar a xxxx. Liguei para xxxx e o xxxx xxxx, e me disseram que a carta de coleta que o XXXX XXXX enviou não era realmente uma carta de coleta, mas uma tentativa de \"alcançar\" para eu fazer um pagamento e que não iria para o meu escritório.   A carta de cobrança foi realmente enviada no XXXX XXXX, 3 dias depois que eles receberam meu pagamento, e eu estava atual na época.   Sinto que esta carta foi uma tentativa de assediar ou me assustar, o que aconteceu. Trabalho duro para manter meu departamento de crédito limpo e essa foi uma maneira muito desagradável de entrar em contato comigo sobre um pagamento com apenas 3 dias de atraso. Além disso, não recebi um telefonema por estar atrasado, mas a carta de cobrança foi automaticamente enviada de acordo com os agentes com os quais discuti o assunto na agência de cobrança e XXXX.   Minha universidade era xxxx xxxx, em xxxx, Indiana\n",
            "Predição Tradicional: Outros\n",
            "Resposta: .    Devo Indianadessa correspondênc\n",
            "======================================================\n",
            "Texto: Tentei comprar um cachorro através do XXXX. Fui solicitado a pagar {$ 800,00} através do XXXX pelo Chase Bank, e o cão seria enviado na segunda -feira seguinte. Paguei esse valor ({$ 800,00}) no sábado, xx/xx/2020 a xxxx xxxx. Este site de criação de cães acabou sendo uma farsa e eu não recebi o cachorro pelo qual paguei. Tentei cancelar a transação na segunda -feira à noite, xx/xx/2020, pois minha conta dizia que a transação ainda estava pendente. Foi-me dito pelo representante que a transação havia sido aprovada e o dinheiro foi depositado em uma conta bancária não perseguida. O representante me disse que a única coisa que eu poderia fazer naquele momento era registrar um relatório policial pelo dinheiro roubado.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Falei com xxxx sobre minhas consultas em xxxx. Eu ainda não os removeu. Falei com eles em xxxx, xxxx e xxxx e eles ainda estão lá. Os credores que divulgaram as consultas por aí enviaram várias cartas para xxxx também. As consultas que precisam ser removidas são as 5 J.P. morgan Chase inquires from : XX/XX/XXXX XX/XX/XXXX XX/XX/XXXX XX/XX/XXXX XX/XX/XXXX 1 inquiry from XXXX on : XX/XX/XXXX XXXX inquiry from XXXX on XX/XX/ Xxxx xxxx consulta de xxxx/xxxx em uma consulta xx/xx/xxxx 1 de xxxx xxxx em xx/xx/xxxx já enviei a documentação afirmando que não fiz essas perguntas.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  \n",
            "\n",
            "Enviei uma recl\n",
            "======================================================\n",
            "Texto: XX/XX/2020: Contactei meu agente de empréstimo de perseguição xxxx xxxx para perguntar sobre as taxas para refinanciar minha residência primária e iniciou o processo de solicitação de refinamento XXXX, 2020: recebeu aprovação condicional no meu aplicativo de refinanciamento. Desde todos os documentos adicionais solicitados pela equipe de empréstimos de perseguição xxxx, 2020: De acordo com o meu provedor de seguros de proprietários de imóveis, o seguro XXXX, eles enviaram os documentos de apólice para perseguir a renovação da apólice de seguro de meus proprietários, mas Chase não pagou na minha conta de custódia xxxx , 2020: Informei meu agente de empréstimo de perseguição xxxx xxxx em uma ligação que não desejo prosseguir com meu aplicativo de refinanciamento e que ele deve cancelá -lo e fechar o arquivo. XXXX XXXX confirmou que ele continuará com o cancelamento do meu aplicativo de refinanciamento.   Chase me enviou uma carta afirmando que o seguro dos meus proprietários de residências expirou e que eles não têm evidências de que eu obtive uma nova cobertura xxxx, 2020: liguei para Chase e pedi que pagassem pelo seguro do meu proprietário pela minha conta de custódia. Eles verificam e me avisam se, como meu pedido de refinanciamento estava ativo, houve uma espera na minha conta de custódia. Portanto, eles não puderam pagar pelo seguro do proprietário da casa da minha conta de custódia. Eu chamo meu agente de empréstimo de perseguição XXXX XXXX e peço que ele certifique -se de que meu aplicativo de refinanciamento esteja fechado para que a retenção na minha conta de custódia possa ser levantada. Xxxx xxxx confirmado por e -mail em xx/xx/2020 que o empréstimo foi cancelado e eu deveria estar bem agora.   Chase me enviou um segundo e último aviso em xx/xx/2020 de que o seguro dos meus proprietários de residências expirou e que eles não têm evidências de que eu obtive uma nova cobertura.   Liguei para o Chase, explicando -os novamente que a retenção do meu garantia deveria ter sido removida agora e eles deveriam pagar imediatamente ao meu provedor de seguros do proprietário da casa, o XXXX Insurance imediatamente. O representante do cliente me disse que ligará para o XXXX Insurance e fará isso do lado deles. Eu os avisei de que existe a possibilidade de que minha política não seja restabelecida porque está atrasada e mesmo que seja restabelecida, pode haver cobranças adicionais que não estou disposto a suportar porque realmente não é minha culpa, meu pagamento deveria ter sido Feito da minha conta de garantia depois que o porão foi levantado. Para o qual o representante do cliente me garantiu que ela reconheceu que é uma supervisão do fim deles e ela o fará enquanto quaisquer cobranças adicionais serão suportadas por eles. Ela ligou para o XXXX Insurance enquanto me mantinha em espera, mas não conseguiu alcançá -los e me pediu para ligar para ela mais tarde para acompanhar. Para ter certeza de que renove o seguro do proprietário da casa imediatamente, liguei para o XXXX Insurance em xx/xx/2020 e conversei com uma pessoa chamada xxxx que me enviou a página de declarações de apólice e confirmou que se Chase o pagar nos próximos 10 dias (Período de carência para credores como Chase), a política pode ser restabelecida.   Liguei para Chase no dia seguinte novamente e perguntei o que XXXX da XXXX me disse e perguntou se eles agora podem seguir em frente e pagar à apólice de seguro do proprietário da minha casa. Chase me pediu para enviar a eles a página de declarações em xxxx xxxx, 2020: Enviei a página de declarações de apólice de seguro do proprietário da casa para xxxx em xx/xx/2020. Parece que o Chase chamou o seguro xxxx em xx/xx/2020, mas não fez nenhum pagamento (aprendi sobre isso depois de entrar em contato com o xxxx xxxx recentemente. Veja o email deles que enviei nos documentos). Recebo outra carta de Chase cerca de duas semanas depois, afirmando que a apólice de seguro dos meus proprietários de casas foi restabelecida.   XXXX, 2020: Recebo uma carta da Chase que o seguro dos meus proprietários de residências expirou e que eles não têm evidências de que obtive uma nova cobertura. Eu pensei que é um erro do fim deles e chamá -los novamente para descobrir o que deu errado. Desta vez, é um agente diferente que não se lembra do que deu errado e não sabe por que minha política não foi restabelecida. Eu digo a eles a história toda e eles novamente reconhecem o erro deles. Eu digo a eles que, devido à culpa do lado deles, eles devem pagar por qualquer valor adicional que incorrer em pagar para obter seguro de outro provedor e eles me colocaram em algum outro departamento para escalar esse assunto. A senhora do outro departamento me diz que não tem o poder de aprovar a quantia adicional se eu tiver que obter outra política de outro provedor. Ela me diz para aumentar o recurso de mensagem segura após o login na minha conta bancária.   XXXX - XXXX, 2020: Começo a escalar o assunto através do recurso de mensagem seguro de perseguições vinculadas à minha conta e, após várias mensagens de acompanhamento, elas me dizem que investigaram o assunto e não acharam que isso foi culpa. Portanto, devo comprar de outro provedor de seguros e qualquer valor adicional não será pago de volta para mim.   Eu compro uma nova cobertura de outro provedor de seguros em {$ 2500,00}, que é mais do que o dobro do valor que eu estava pagando originalmente ({$ 1200,00}) a xxxx xxxx.   Agora estou entrando em contato com o CFPB para me ajudar a obter o valor adicional que paguei ({$ 1300,00}) da Chase ou faça com que eles comprem seguro de proprietários de casas para minha casa às custas deles, para não ter que pagar esse valor exorbitante por nenhuma culpa minha.   Espero que você ajude a resolver esse assunto com o mais cedo possível. Anexei os documentos para fornecer visibilidade total ao que aconteceu. Também informei meu agente de empréstimo de perseguição XXXX XXX sobre isso e disse a ele que irei levantando isso como uma reclamação com o CFPB.   Deixe -me saber se você tiver alguma dúvida.   Obrigado!\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Recentemente, tentei refinanciar minha hipoteca. Eu tenho um empréstimo apoiado pela FHA com um número do caso da FHA. Chase me disse que o empréstimo não era um empréstimo da FHA, mas enviei a documentação mostrando que tinha um caso de FHA # anexado à minha declaração anos após o fechamento do empréstimo. Eu também tinha o HUD-1 que provou ser o mesmo.   Os agentes de empréstimos começaram a me dizer que estava mostrando a FHA, mas não conseguiram encontrar o número do caso. Eu tenho recebido PMI de cobrança no empréstimo da FHA desde xx/xx/xxxx, mas eles não conseguiram refinanciar como FHA. Em vez de investigar mais, os agentes de empréstimos acabaram de repudiar o empréstimo. Os oficiais de empréstimos são xxxx xxxx nmls id # xxxx e xxxx xxxx.   Por favor, veja este caso. Antes de entrar em contato com um advogado sobre isso, gostaria de ver se o Chase pode resolver isso internamente. Parece haver algo duvidoso sobre isso para mim.   Atenciosamente, xxxx xxxx\n",
            "Predição Tradicional: Hipotecas / Empréstimos\n",
            "Resposta: \n",
            "======================================================\n",
            "Texto: Usei o XXXX no meu aplicativo Chase QuickPay para pagar a um endereço de e -mail de um dos meus contratados. Quando enviei o pagamento, ele não foi recebido na conta pretendida que pertence ao meu contratado, mas em outra conta. Quando recebi uma solicitação do mesmo contratado com a mesma conta de email, o dinheiro chegou à conta correta. Assim, eu tinha duas transações, uma por {$ 130,00} que foi para a conta errada e uma de {$ 130,00} que foi para a conta certa. Eu arquivei uma reclamação na Chase, mas eles negaram porque iniciei a transferência. Quando tentei contestá -lo, fui a corrida com o atendimento ao cliente.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  Eu não sei como prosse\n",
            "======================================================\n",
            "Texto: Eu tenho um cartão Chase XXXX XXXX que obtive para os benefícios de viagem, que incluem resgate de pontos em xxxx % se resgatados usando o site da Chase Travel. Eu tenho tentado reservar viagens no site da Chase Travel desde pelo menos xx/xx/2019, mas não consigo resgatar meus pontos. Depois de muitas ligações para perseguir ao longo de semanas, soube que o Chase está ciente, pois pelo menos o XX/XX/2019, de um problema técnico que impede que pelo menos alguns detentores de cartões de reserva de resgatam pontos no site de viagens. Fiz muitas tentativas para resolver esse problema com o Chase. Disseram -me que a única opção disponível é fazer viajar para outro lugar e solicitar um crédito de extrato (essencialmente em dinheiro de volta) da Chase, que será um dólar por crédito em dólares. O Chase não honrará o valor de resgate do XXXX % oferecido contratual para viagens, embora eu tenha feito essa solicitação em várias ocasiões. Chase não pode ou não fornecer um ETA sobre quando esse assunto será resolvido.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Estou muito frustrado pela\n",
            "======================================================\n",
            "Texto: Liguei para Chase e disse a eles que minha conta foi roubada e meus cartões foram perdidos e, enquanto tentava fazer login no nome de usuário e a senha mudou, o email e o número de telefone mudaram como eu disse a eles que me disseram que minha conta foi fechado e não tem mais conta com eles, pois foi restrito e eles disseram que não terei acesso a ele\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  e teria que criar um novo\n",
            "======================================================\n",
            "Texto: Em XX/XX/2019, recebi uma pilha de correio do Chase Banking, com quem nunca fiz negócios e não pretendia. Duas peças deste e -mail inesperado foram ativar novos cartões de débito para duas contas de corrente fraudulenta abertas em meu nome sem minha autorização. Três peças foram retornadas devolvidas datadas de xx/xx/2019 de xxxx xxxx xxxx (roteamento # xxxx), das quais nunca ouvi falar eletronicamente depositado nas contas fraudulentas nos valores da verificação # xxxx {$ 900.00}, # xxxx {$ 750.00}} e verifique # xxxx {$ 800.00}. Três peças foram retornadas devolvidas datadas de xx/xx/2019 depositadas nas contas de corrente fraudulenta do xxxx xxxx xxxx xxxx, xxxx (roteamento # xxxx), com quem eu tenho uma política de deficiência opcional, as verificações estavam na quantidade de verificação # xxxx {{ $ 1800.00}, cheque # xxxx {$ 1900.00} e verifique # xxxx {$ 1600.00}. Duas peças de correio afirmaram que essas contas bancárias do Chase haviam sido bloqueadas e logo fechariam. Não tenho confirmação de que essas contas foram encerradas e nenhuma informação adicional me foi dada sobre o pedido para abrir essas contas ou quaisquer depósitos/retiradas adicionais. Não tenho números de reclamação para essas empresas porque isso não foi autorizado por mim. Estudei meu relatório de crédito e encontrei inquéritos não autorizados recentemente e endereços fraudulentos.\n",
            "Predição Tradicional: Serviços de conta bancária\n",
            "Resposta:  Estou pedindo um relatório comple\n",
            "======================================================\n",
            "Texto: Arquivei uma disputa de qualidade de serviço em xx/xx/xxxx (disputa # xxxx). A investigação não foi realizada corretamente pelo banco e, portanto, o banco pediu que eu refile a disputa. Eu o arquivei na segunda vez em xx/xx/xxxx (disputa # xxxx). Em uma carta datada de xx/xx/xxxx, o banco disse que as cobranças são válidas com base na informações fornecidas pelo comerciante. Pedi ao banco que me fornecesse uma cópia das informações recebidas do comerciante. Disseram -me que a informação não foi recebida do comerciante, pois é voluntária. Com base nas informações conflitantes fornecidas, entrei em contato com o Escritório Executivo do Banco através do BBB. Em xx/xx/xxxx, fui informado pelo Escritório Executivo de que eles não podem registrar uma disputa devido às diretrizes fornecidas pelo XXXX. Em xx/xx/xxxx, entrei em contato com xxxx e fui informado de que a disputa pode ser arquivada sob o código da razão 53.   Chase não havia repetidamente contestar a acusação com o comerciante. Ao fazer isso, Chase está atrás de comerciantes fraudulentos.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta:  Isso faz com que a disputa\n",
            "======================================================\n",
            "Texto: Os serviços do Chase Cardmember continuam a relatar informações imprecisas no meu relatório de crédito. Eu os notifiquei que essa conta não me pertence. Eles se recusam a investigar e verificar para mostrar a prova de que essa conta não me pertence. Solicitei que eles removam este item do meu relatório de crédito, mas Chase se recusou a fazê -lo. Este relatório no meu relatório de crédito prejudicou meu arquivo de crédito completamente. Não há como recuperar os pontos que perdi devido à negligência de relatórios imprecisos. Chase continua a violar meus direitos sob a Lei de Relatórios de Crédito Justo relatando informações imprecisas e danificando meu arquivo de crédito. Incluí uma cópia do meu recente relatório de crédito XXXX para mostrar os relatórios adversos no meu relatório de crédito. Como você verá, todos os credores estão em excelentes serviços de membro do cartão Chase continuam a relatar informações imprecisas no meu relatório de crédito. Eu os notifiquei que essa conta não me pertence. Eles se recusam a investigar e verificar para mostrar a prova de que essa conta não me pertence. Solicitei que eles removam este item do meu relatório de crédito, mas Chase se recusou a fazê -lo. Este relatório no meu relatório de crédito prejudicou meu arquivo de crédito completamente. Não há como recuperar os pontos que perdi devido à negligência de relatórios imprecisos. Chase continua a violar meus direitos sob a Lei de Relatórios de Crédito Justo relatando informações imprecisas e danificando meu arquivo de crédito. Incluí uma cópia do meu recente relatório de crédito XXXX para mostrar os relatórios adversos no meu relatório de crédito. Como você verá, todos os credores estão em excelente posição. Sou extremamente responsável com meu crédito.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  Por favor, remova imediatamente\n",
            "======================================================\n",
            "Texto: Atualmente, sou divorciado e o principal custodiante de minhas filhas XXXX. Estou em processo de perda de nossa casa devido a problemas de crédito específicos que estão me impedindo de garantir um empréstimo de refinanciamento e diminuir nossas despesas mensais de moradia. Durante o meu casamento, houve várias contas abertas de forma fraudulenta usando meu número de seguridade social, sem o conhecimento de mim, pela minha então esposa. Tentei alcançar esses credores e nunca recebi as chamadas. Eles foram acusados ​​e agora estão afetando a mim e às minhas filhas, em vez da pessoa que originalmente abriu essas contas. Estou pedindo ajuda para remover essas contas específicas do meu relatório de crédito para que eu possa garantir nossa casa e as necessidades futuras. obrigada\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: .\n",
            "======================================================\n",
            "Texto: Na segunda -feiraxx/xx/xxxx, encontrei um add para xxxx tickets no xxxx e eu mandei uma mensagem para o vendedor que passou por xxxx xxxx, ele me disse que estava disposto a me vender um ingresso por {$ 240,00} se eu estivesse disposto a usar xxxx. Hoje, xx/xx/xxxx, eu mando texto para o vendedor novamente, recebo as informações de pagamento e decidimos sobre um ponto de encontro para que pudéssemos iniciar a transação, recebi o endereço de e -mail dele xxxx, comecei a enviar o pagamento e me disseram que isso Ele estará no local de encontro em 45 minutos. Ele me manda uma mensagem alegando que está entrando e bloqueia meu número de telefone para que eu não possa entrar em contato com ele.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "A resposta que eu forne\n",
            "======================================================\n",
            "Texto: O Chase Bankad reivindica o empréstimo de 48 meses em 2,38% para verificar a conta de conta patronsonly, capaz de aplicar o aplicativo onlineOnline \"não funciona\" ou \"desaparece\" on -line com um agente não ajuda. (Obter um agente é um evento hercúleo) em qualquer caso, agentes on -line e, na filial, garante que eu nunca receberei os 2,38% de qualquer maneira: \"ninguém faz\", sou um cliente de longa data da Chase e sempre tive um Excelente pontuação de crédito. Não é essa trapaça?\n",
            "Predição Tradicional: Outros\n",
            "Resposta:  Inadequado? Burlar ?\n",
            "\n",
            "Sim\n",
            "======================================================\n",
            "Texto: Sou vítima de roubo de identidade. Há vários anos, alguém invadiu meu escritório e roubou vários documentos de identificação, incluindo meu cartão de segurança social, carteira de motorista e passaporte. Entrei com um relatório policial no Departamento de Polícia XXXX. O XXXX aprovou meu relatório e me deu o número do relatório oficial # Número do relatório: XXXX Anexei uma cópia do relatório a esta reclamação.   Ao verificar meu crédito, encontrei um item de \"Chase\" para um empréstimo de automóvel. Ao fazer algumas escavações, descobri que esse empréstimo foi obtido de uma concessionária de automóveis.   Eu nunca fiz negócios com esta concessionária.   Várias contas bancárias foram abertas em meu nome e costumavam pagar por vários empréstimos fraudulentos, também realizados em meu nome. Este empréstimo não foi autorizado ou usado por mim. Não recebi ganho pessoal com este empréstimo.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta: \n",
            "\n",
            "Este empréstimo fraudulent\n",
            "======================================================\n",
            "Texto: Xx/xx/xxxx e xxxx, 2019. Meu número de cartão de crédito foi usado de forma fraudulenta para fazer {$ 2100.00} em compras. Segundo Chase, o comprador tinha um cartão de chip no ponto de venda. Minha família possuía todos os meus cartões. Chase entrou em contato comigo duas vezes e as duas vezes eu disse a eles que as cobranças listadas não eram minhas, exceto pela carga xxxx. Eles reembolsaram as acusações na minha conta. Então, dois meses depois, eles enviaram uma carta afirmando que eu me beneficiei de algumas dessas cobranças e me revoguei {$ 1700,00} para minha conta. Eles declararam que a investigação estava completa. Eles nunca me questionaram na investigação. Podemos provar que nenhum de nós que carrega um dos cartões estava no local geográfico onde as compras foram feitas. Existem até cobranças corretas feitas nos dois dias por mim, que mostram que eu não estava no xxxx xxxx. Eles também afirmam que eu os notifiquei que uma das compras XXXX foi minha que eu não fiz e a acusação não era minha. Chase não me incluiu em sua investigação. Eu posso provar que nenhum da minha família estava no local em que as acusações foram feitas. Considero que Chase está agindo de má fé.\n",
            "Predição Tradicional: Roubo / Relatório de disputa\n",
            "Resposta: \n",
            "\n",
            "Segundo as regras de má\n",
            "======================================================\n",
            "Texto: Recebi uma oferta promocional endereçada a mim pessoalmente do Chase Bank para o seu cartão xxxx xxxx xxxx xxxx xxxx que oferecia xxxx bônus milhas na minha conta de panfleto frequente XXXX se eu solicitei um cartão e gastei {$ 3000,00} nos primeiros 3 meses de uso e e e mais xxxx milhas adicionais se eu adicionar um usuário autorizado. Eu me inscrevi e recebi o cartão, mas me disseram que a promoção só me daria xxxx milhas. Reclamei com o Chase Bank no XXXX # na parte traseira do cartão e recebi uma carta datada de xxxx xxxx, 2017 de xxxx xxxx, especialista em atendimento ao cliente dizendo que eles não poderiam corrigir o problema e só posso receber xxxxxx xxx. Liguei de novo e falei com um supervisor e me disseram a mesma coisa. Esta é uma isca e interruptor simples. Eles me ofereceram xxxx milhas, eu aceitei a oferta e agora só me darão xxxx milhas. Liguei duas vezes para que eles saibam do erro e eles disseram que não podiam fazer nada a respeito.\n",
            "Predição Tradicional: Cartão de crédito / Cartão pré-pago\n",
            "Resposta:  Por favor, ajude-me a ob\n",
            "======================================================\n"
          ]
        }
      ],
      "source": [
        "for index, prompt in test_data.items():\n",
        "    traditional_prediction = nb_classifier.predict(tfidf_vectorizer.transform([prompt]))[0]\n",
        "    #prompt = \"Colocar um texto\"\n",
        "    gpt3_response = generate_text(prompt)\n",
        "\n",
        "    print(f\"Texto: {prompt}\")\n",
        "    print(f\"Predição Tradicional: {traditional_prediction}\")\n",
        "    print(f\"Resposta: {gpt3_response}\")\n",
        "    print(\"======================================================\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "L3yZOy9agWzQ"
      },
      "source": [
        "# Exemplo de uso fora o CSV"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 38,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mD7yNI9pDZQ6",
        "outputId": "1bc32718-735c-456c-e2c4-bbe0290505c9"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Reclamação gerada: \n",
            "\n",
            "1. Avenida Paulista\n",
            "2\n"
          ]
        }
      ],
      "source": [
        "prompt = \"Nomes de rua famosas em São paulo\"\n",
        "generated_text = generate_text(prompt)\n",
        "print(\"Reclamação gerada:\", generated_text)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7nn1FIiGmdNz"
      },
      "source": [
        "# **Parte 3 (Extra) – Utilizar a IA Generativa para fazer uma classificação livre de assuntos e avaliar qualitativamente os resultados.**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 115,
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Importação de bibliotecas\n",
        " \n",
        "import openai\n",
        "import os\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 116,
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# definição e uso da base de dados \n",
        "url = \"https://raw.githubusercontent.com/thiagonogueira/datasets/main/tickets_reclamacoes_classificados_one_line_generative.csv\"\n",
        "df = pd.read_csv(url, delimiter=';')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 117,
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "#chave da API Openai\n",
        "openai.api_key = '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'"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 147,
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "#Comando para openai classificar e resumir as reclamações em no maximo tres palavras\n",
        "\n",
        "\n",
        "def analyze_complaints(complaints):\n",
        "    results = []\n",
        "\n",
        "    for complaint in complaints:\n",
        "        completion = openai.Completion.create(\n",
        "            engine=\"text-davinci-002\",\n",
        "            prompt=f\"Classificar e resumir em três palavras o motivo da reclamação do cliente:\\n'{complaint}'\",\n",
        "            max_tokens=50,\n",
        "            temperature=0.5,\n",
        "            stop=None\n",
        "        )\n",
        "        \n",
        "        response = completion['choices'][0]['text'].strip()\n",
        "\n",
        "        results.append({\n",
        "            'Reclamação': complaint,\n",
        "            'Classificacao_Motivo': response\n",
        "        })\n",
        "\n",
        "    return results\n",
        "\n",
        "#utilizado o nome 'complaint' para não confundir com nome de coluna na base de dados ou outras análises"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 148,
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "#define o range de análise para 50 linhas da base\n",
        "complaints_50 = [\"Reclamação \" + str(i) for i in range(1, 51)]\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 149,
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Analisa as linhas da base\n",
        "results_50 = analyze_complaints(complaints_50)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 150,
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Reclamação: Reclamação 1\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "\n",
            "Incorrect billing.\n",
            "=====================================\n",
            "Reclamação: Reclamação 2\n",
            "Classificação do Motivo: Preço alto.\n",
            "=====================================\n",
            "Reclamação: Reclamação 3\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 4\n",
            "Classificação do Motivo: Não recebido.\n",
            "=====================================\n",
            "Reclamação: Reclamação 5\n",
            "Classificação do Motivo: Mau atendimento.\n",
            "=====================================\n",
            "Reclamação: Reclamação 6\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 7\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 8\n",
            "Classificação do Motivo: Fatura alta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 9\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 10\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 11\n",
            "Classificação do Motivo: Fatura alta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 12\n",
            "Classificação do Motivo: Irregularidades na entrega.\n",
            "=====================================\n",
            "Reclamação: Reclamação 13\n",
            "Classificação do Motivo: Incompetência, falta de atendimento, falta de soluções.\n",
            "=====================================\n",
            "Reclamação: Reclamação 14\n",
            "Classificação do Motivo: Cliente insatisfeito.\n",
            "=====================================\n",
            "Reclamação: Reclamação 15\n",
            "Classificação do Motivo: Preço alto.\n",
            "=====================================\n",
            "Reclamação: Reclamação 16\n",
            "Classificação do Motivo: Pagamento atrasado.\n",
            "=====================================\n",
            "Reclamação: Reclamação 17\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 18\n",
            "Classificação do Motivo: Resumo:\n",
            "\n",
            "1. O cliente está insatisfeito com o atendimento recebido;\n",
            "2. O cliente acha que o problema não foi resolvido;\n",
            "=====================================\n",
            "Reclamação: Reclamação 19\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 20\n",
            "Classificação do Motivo: - Problema na entrega\n",
            "- Atraso na entrega\n",
            "=====================================\n",
            "Reclamação: Reclamação 21\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 22\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 23\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 24\n",
            "Classificação do Motivo: Pagamento, cobrança, valor.\n",
            "=====================================\n",
            "Reclamação: Reclamação 25\n",
            "Classificação do Motivo: Cliente insatisfeito.\n",
            "=====================================\n",
            "Reclamação: Reclamação 26\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 27\n",
            "Classificação do Motivo: Não recebido.\n",
            "=====================================\n",
            "Reclamação: Reclamação 28\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 29\n",
            "Classificação do Motivo: Falta de estoque.\n",
            "=====================================\n",
            "Reclamação: Reclamação 30\n",
            "Classificação do Motivo: Cliente não satisfeito.\n",
            "=====================================\n",
            "Reclamação: Reclamação 31\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 32\n",
            "Classificação do Motivo: Atraso no envio.\n",
            "=====================================\n",
            "Reclamação: Reclamação 33\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 34\n",
            "Classificação do Motivo: Fatura alta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 35\n",
            "Classificação do Motivo: Falta de produto.\n",
            "=====================================\n",
            "Reclamação: Reclamação 36\n",
            "Classificação do Motivo: Fatura alta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 37\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 38\n",
            "Classificação do Motivo: Cliente insatisfeito.\n",
            "=====================================\n",
            "Reclamação: Reclamação 39\n",
            "Classificação do Motivo: Fatura alta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 40\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 41\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 42\n",
            "Classificação do Motivo: Falta de peças.\n",
            "=====================================\n",
            "Reclamação: Reclamação 43\n",
            "Classificação do Motivo: Fatura incorreta.\n",
            "=====================================\n",
            "Reclamação: Reclamação 44\n",
            "Classificação do Motivo: Pagamento, atrasado, cobrança.\n",
            "=====================================\n",
            "Reclamação: Reclamação 45\n",
            "Classificação do Motivo: Atraso, falta de peças, má qualidade.\n",
            "=====================================\n",
            "Reclamação: Reclamação 46\n",
            "Classificação do Motivo: Cobrança indevida.\n",
            "=====================================\n",
            "Reclamação: Reclamação 47\n",
            "Classificação do Motivo: Falta de produto.\n",
            "=====================================\n",
            "Reclamação: Reclamação 48\n",
            "Classificação do Motivo: Fatura, cobrança, valor.\n",
            "=====================================\n",
            "Reclamação: Reclamação 49\n",
            "Classificação do Motivo: Cliente insatisfeito.\n",
            "=====================================\n",
            "Reclamação: Reclamação 50\n",
            "Classificação do Motivo: Fatura alta.\n",
            "=====================================\n"
          ]
        }
      ],
      "source": [
        "# Apresenta o resultado da análise\n",
        "for result in results_50:\n",
        "    print(f\"Reclamação: {result['Reclamação']}\")\n",
        "    print(f\"Classificação do Motivo: {result['Classificacao_Motivo']}\")\n",
        "    print(\"=====================================\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 180,
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "image/png": "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",
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "#Prepara o resultado para apresentação gráfica em formato de barras - top 10\n",
        "\n",
        "resumos_50 = [result['Classificacao_Motivo'] for result in results_50]\n",
        "resumos_counts = {resumo: resumos_50.count(resumo) for resumo in set(resumos_50)}\n",
        "sorted_resumos_counts = sorted(resumos_counts.items(), key=lambda x: x[1], reverse=True)[:10]\n",
        "\n",
        "resumos, counts = zip(*sorted_resumos_counts[::-1])\n",
        "plt.barh(resumos, counts, color='orange')\n",
        "plt.xlabel('Número de Ocorrências')\n",
        "plt.ylabel('Resumo do Motivo da Reclamação')\n",
        "plt.title('Resumo dos Motivos de Reclamação')\n",
        "plt.xticks(rotation=45, ha='right')  \n",
        "plt.yticks(fontsize=6)  \n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "Avaliação qualitativa com Open Ai"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 182,
      "metadata": {},
      "outputs": [
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "  <style>\n",
              "    pre {\n",
              "        white-space: pre-wrap;\n",
              "    }\n",
              "  </style>\n",
              "  "
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# É possível notar que a classificação gerada por openai/generativa apresenta \n",
        "# resultados mais qualitativos, possibilitando tomada de ação em causa raiz de problemas\n",
        "# caso o modelo esteja bem treinado.\n",
        "\n",
        "# Fatura incorreta/ Fatura alta apresentam mais facilidade ao tratar o problema pontual do cliente\n",
        "# ou até mesmo problemas massivos que não seriam identificados facilmente\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
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
      "version": "3.11.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
