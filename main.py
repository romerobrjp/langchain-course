from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")
    # print(os.getenv("OPENAI_API_KEY"))
    information = """
        Warcraft is a fantasy video game series and media franchise created by Blizzard Entertainment. The series consists of six core games: Warcraft: Orcs & Humans (1994), Warcraft II: Tides of Darkness (1995), Warcraft III: Reign of Chaos (2002), World of Warcraft (2004), Hearthstone (2014), and Warcraft Rumble (2023). Initially a real-time strategy (RTS) series, Warcraft expanded into other game genres beginning with World of Warcraft, a highly influential massively multiplayer online role-playing game (MMORPG). The franchise has also spawned novels, comics, a tabletop role-playing game, a trading card game, and a 2016 feature film.

        The franchise is primarily set on the planet Azeroth, as well as related planets and metaphysical dimensions. Azeroth is inhabited by various races and civilizations, including typical fantasy races such as elves, dwarves, gnomes, orcs, and trolls, along with original races and creatures unique to the franchise. Its lore and story center on warfare between the races and factions of Azeroth, typically between the human-led Alliance and the orc-led Horde, chronicling the exploits of heroes and villains on both sides.[3][4] While high fantasy at its core, the Warcraft universe incorporates a diverse assortment of influences, including science fiction and dark fantasy. Warcraft has been noted as differentiating itself from other fantasy universes by highlighting "monster races" such as orcs, trolls, and undead, often portraying them as protagonists and giving them significant character development and moral complexity.

        The Warcraft franchise has been highly successful, grossing over $12 billion in revenue, making it one of the highest-grossing video game franchises of all time.[5] The games have been critically acclaimed: the first three Warcraft games are considered landmarks of the RTS genre, while World of Warcraft is regarded as the most popular and influential MMORPG of all time
    """
    summary_template = """
        given the information {information} about a game, I want you to create:
        1. a short summary
        2. two interesting facts about the game
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )
    # llm = ChatOpenAI(model="gpt-4.1", temperature=0) # temperature set to 0 for deterministic output. as higher temperature increases randomness/creativity
    # llm = ChatOpenAI(model="gpt-5")
    llm = ChatOllama(temperature=0, model="mistral")
    chain = summary_prompt_template | llm # the pipe operator will create a chain that first formats the prompt and then passes it to the llm
    response = chain.invoke(input={"information": information}) # invoke the chain with the input information
    print(response.content)

if __name__ == "__main__":
    main()
