import asyncio
import os

from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
from poke_env.player import RandomPlayer

from src.battle_agent import BattleAgent

load_dotenv()

BATTLE_FORMAT = "gen9randombattle"
N_BATTLES = 1


async def main():
    client = wrap_openai(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    agent = BattleAgent(
        openai_client=client,
        model="gpt-4o-mini",
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
    )
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
    )

    print(f"Starting {N_BATTLES} battle(s): BattleAgent vs RandomPlayer")
    print(f"Format: {BATTLE_FORMAT}")
    print(f"Watch at: http://localhost:8000\n")

    await agent.battle_against(opponent, n_battles=N_BATTLES)

    print(f"\nResults: {agent.n_won_battles}W / "
          f"{agent.n_finished_battles - agent.n_won_battles}L "
          f"({agent.n_finished_battles} total)")


if __name__ == "__main__":
    asyncio.run(main())
