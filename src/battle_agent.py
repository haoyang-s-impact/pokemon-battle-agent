import json
from typing import Optional

from langsmith import traceable
from openai import AsyncOpenAI
from poke_env.battle import Battle, Move, Pokemon
from poke_env.player import Player

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "choose_move",
            "description": "Choose an attack or status move to use this turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "move_name": {
                        "type": "string",
                        "description": "The name/id of the move to use (e.g. 'thunderbolt', 'stealthrock').",
                    }
                },
                "required": ["move_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "choose_switch",
            "description": "Switch to a different Pokemon on your team.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pokemon_name": {
                        "type": "string",
                        "description": "The species name of the Pokemon to switch to (e.g. 'gengar', 'starmie').",
                    }
                },
                "required": ["pokemon_name"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are a Pokemon battle agent playing a random battle. Each turn you must \
choose exactly one action: use a move or switch to another Pokemon.

You receive the current battle state including your active Pokemon, the \
opponent's active Pokemon, your available moves with their stats, and your \
available switches. Use your knowledge of Pokemon type matchups, move \
categories, and battle strategy to make the best decision.

Always respond with a single tool call: either choose_move or choose_switch.\
"""


def _normalize(name: str) -> str:
    return "".join(c for c in name if c.isalnum()).lower()


class BattleAgent(Player):
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        model: str = "gpt-4o-mini",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = openai_client
        self.model = model

    def _format_battle_state(self, battle: Battle) -> str:
        active = battle.active_pokemon
        active_info = (
            f"Your active Pokemon: {active.species} "
            f"(Type: {'/'.join(str(t) for t in active.types if t)}) "
            f"HP: {active.current_hp_fraction * 100:.1f}% "
            f"Status: {active.status.name if active.status else 'None'} "
            f"Boosts: {active.boosts}"
        )

        opp = battle.opponent_active_pokemon
        if opp:
            opp_info = (
                f"Opponent's active Pokemon: {opp.species} "
                f"(Type: {'/'.join(str(t) for t in opp.types if t)}) "
                f"HP: {opp.current_hp_fraction * 100:.1f}% "
                f"Status: {opp.status.name if opp.status else 'None'} "
                f"Boosts: {opp.boosts}"
            )
        else:
            opp_info = "Opponent's active Pokemon: Unknown"

        moves_info = "Available moves:\n"
        if battle.available_moves:
            for m in battle.available_moves:
                moves_info += (
                    f"- {m.id} (Type: {m.type}, BP: {m.base_power}, "
                    f"Acc: {m.accuracy}, PP: {m.current_pp}/{m.max_pp}, "
                    f"Cat: {m.category.name})\n"
                )
        else:
            moves_info += "- None (must switch)\n"

        switches_info = "Available switches:\n"
        if battle.available_switches:
            for p in battle.available_switches:
                switches_info += (
                    f"- {p.species} (Type: {'/'.join(str(t) for t in p.types if t)}, "
                    f"HP: {p.current_hp_fraction * 100:.1f}%, "
                    f"Status: {p.status.name if p.status else 'None'})\n"
                )
        else:
            switches_info += "- None\n"

        return (
            f"{active_info}\n"
            f"{opp_info}\n\n"
            f"{moves_info}\n"
            f"{switches_info}\n"
            f"Weather: {battle.weather}\n"
            f"Terrains: {battle.fields}\n"
            f"Your side conditions: {battle.side_conditions}\n"
            f"Opponent side conditions: {battle.opponent_side_conditions}"
        )

    def _find_move_by_name(self, battle: Battle, move_name: str) -> Optional[Move]:
        normalized = _normalize(move_name)
        for move in battle.available_moves:
            if move.id == normalized:
                return move
        for move in battle.available_moves:
            if _normalize(move.name) == normalized:
                return move
        return None

    def _find_pokemon_by_name(
        self, battle: Battle, pokemon_name: str
    ) -> Optional[Pokemon]:
        normalized = _normalize(pokemon_name)
        for pkmn in battle.available_switches:
            if _normalize(pkmn.species) == normalized:
                return pkmn
        return None

    @traceable(name="choose_move")
    async def choose_move(self, battle: Battle) -> str:
        battle_state = self._format_battle_state(battle)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": battle_state},
                ],
                tools=TOOL_SCHEMA,
                tool_choice="required",
            )

            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                fn_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if fn_name == "choose_move":
                    move_name = args.get("move_name", "")
                    move = self._find_move_by_name(battle, move_name)
                    if move and move in battle.available_moves:
                        print(f"  -> Using move: {move.id}")
                        return self.create_order(move)
                    print(f"  !! Invalid move '{move_name}', falling back to random")

                elif fn_name == "choose_switch":
                    pkmn_name = args.get("pokemon_name", "")
                    pkmn = self._find_pokemon_by_name(battle, pkmn_name)
                    if pkmn and pkmn in battle.available_switches:
                        print(f"  -> Switching to: {pkmn.species}")
                        return self.create_order(pkmn)
                    print(
                        f"  !! Invalid switch '{pkmn_name}', falling back to random"
                    )
                else:
                    print(f"  !! Unknown tool '{fn_name}', falling back to random")
            else:
                print("  !! No tool call in response, falling back to random")

        except Exception as e:
            print(f"  !! OpenAI error: {e}, falling back to random")

        return self.choose_random_move(battle)
