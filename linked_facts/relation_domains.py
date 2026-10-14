"""Relation domains mixed into each linked-facts world.

Agent 1 states the person-to-middle hop and agent 2 states the middle-to-answer
hop. A world cycles through membership, color, animal, furniture, and pledged
group so the text is not only membership and headquarters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


PROPER_NAME_SOURCES = frozenset({"groups", "locations"})

MEMBERSHIP_KINDS = (
    "alliance",
    "circle",
    "collective",
    "company",
    "crew",
    "fellowship",
    "guild",
    "league",
    "society",
    "team",
    "unit",
)

PLEDGED_GROUP_KINDS = (
    "house",
    "order",
    "clan",
    "choir",
    "council",
    "lodge",
    "chapter",
    "troupe",
    "brigade",
    "pack",
    "school",
    "band",
)

LEXICONS: dict[str, dict[str, tuple[str, ...]]] = {
    "colors": {
        "familiar": (
            "amber",
            "crimson",
            "indigo",
            "violet",
            "scarlet",
            "cobalt",
            "saffron",
            "emerald",
            "magenta",
            "turquoise",
            "lavender",
            "ochre",
            "maroon",
            "ivory",
            "copper",
            "bronze",
        ),
        "held_out": (
            "cerulean",
            "burgundy",
            "mustard",
            "charcoal",
            "apricot",
            "fuchsia",
            "viridian",
            "periwinkle",
            "chartreuse",
            "vermilion",
            "ultramarine",
            "carmine",
            "celadon",
            "russet",
            "alabaster",
            "amethyst",
        ),
    },
    "coat_animals": {
        "familiar": (
            "pelican",
            "penguin",
            "walrus",
            "buffalo",
            "cheetah",
            "dolphin",
            "gazelle",
            "gorilla",
            "hamster",
            "jaguar",
            "leopard",
            "manatee",
            "meerkat",
            "octopus",
            "panther",
            "sparrow",
        ),
        "held_out": (
            "albatross",
            "armadillo",
            "chameleon",
            "crocodile",
            "flamingo",
            "hedgehog",
            "kangaroo",
            "mongoose",
            "ostrich",
            "platypus",
            "porcupine",
            "salamander",
            "tortoise",
            "woodpecker",
            "antelope",
            "elephant",
        ),
    },
    "kept_animals": {
        "familiar": (
            "badger",
            "falcon",
            "heron",
            "iguana",
            "alpaca",
            "bison",
            "coyote",
            "ferret",
            "gibbon",
            "lobster",
            "mussel",
            "pigeon",
            "python",
            "turtle",
            "vulture",
            "starfish",
        ),
        "held_out": (
            "chipmunk",
            "giraffe",
            "axolotl",
            "barracuda",
            "caribou",
            "echidna",
            "halibut",
            "jackal",
            "kestrel",
            "lamprey",
            "narwhal",
            "tapir",
            "wallaby",
            "wombat",
            "bobcat",
            "cougar",
        ),
    },
    "rest_furniture": {
        "familiar": (
            "armchair",
            "bookcase",
            "credenza",
            "daybed",
            "dresser",
            "footstool",
            "hammock",
            "loveseat",
            "mattress",
            "nightstand",
            "ottoman",
            "recliner",
            "sideboard",
            "wardrobe",
            "workbench",
            "headboard",
        ),
        "held_out": (
            "chifforobe",
            "davenport",
            "escritoire",
            "fourposter",
            "hallstand",
            "hopechest",
            "lampstand",
            "pierglass",
            "plantstand",
            "secretaire",
            "teacart",
            "tallboy",
            "washbowl",
            "wingchair",
            "footlocker",
            "blanketchest",
        ),
    },
    "owned_furniture": {
        "familiar": (
            "armoire",
            "bookshelf",
            "bunkbed",
            "commode",
            "cradle",
            "divan",
            "glider",
            "hutch",
            "lounger",
            "rocker",
            "settee",
            "trestle",
            "vanity",
            "washstand",
            "highchair",
            "bassinet",
        ),
        "held_out": (
            "canopybed",
            "coatrack",
            "coffeetable",
            "diningchair",
            "endtable",
            "filingcabinet",
            "papasan",
            "shoerack",
            "storagebench",
            "windowseat",
            "winecabinet",
            "worktable",
            "writingtable",
            "campstool",
            "lectern",
            "hatrack",
        ),
    },
    "rooms": {
        "familiar": (
            "attic",
            "cellar",
            "parlor",
            "gallery",
            "nursery",
            "pantry",
            "library",
            "chapel",
            "stable",
            "workshop",
            "cloister",
            "armory",
            "solarium",
            "vestibule",
            "alcove",
            "refectory",
        ),
        "held_out": (
            "antechamber",
            "ballroom",
            "bedchamber",
            "cloakroom",
            "courtyard",
            "maproom",
            "pottingroom",
            "guardroom",
            "mudroom",
            "playroom",
            "chartroom",
            "stockroom",
            "sunroom",
            "taproom",
            "wardroom",
            "washroom",
        ),
    },
}


@dataclass(frozen=True)
class RelationDomain:
    """One two-hop relation that can appear inside a linked-facts world."""

    domain_id: str
    middle_source: str
    answer_source: str
    kinds: tuple[str, ...]
    agent_1_bindings: tuple[tuple[str, str], ...]
    agent_2_bindings: tuple[tuple[str, str], ...]
    agent_1_templates: Mapping[str, tuple[tuple[str, str], ...]]
    agent_2_templates: Mapping[str, tuple[tuple[str, str], ...]]
    question_templates: Mapping[str, tuple[tuple[str, str], ...]]

    def agent_1_arguments(self, person: str, middle: str, kind: str) -> dict[str, str]:
        """Bind placeholders for this domain's person-to-middle sentence."""
        return self._bind(self.agent_1_bindings, person, middle, kind, "")

    def agent_2_arguments(self, middle: str, kind: str, answer: str) -> dict[str, str]:
        """Bind placeholders for this domain's middle-to-answer sentence."""
        return self._bind(self.agent_2_bindings, "", middle, kind, answer)

    def question_arguments(self, person: str) -> dict[str, str]:
        """Bind the question, which names only the person."""
        return {"person": person}

    def _bind(
        self,
        bindings: tuple[tuple[str, str], ...],
        person: str,
        middle: str,
        kind: str,
        answer: str,
    ) -> dict[str, str]:
        """Map placeholder names onto this fact's person, middle, kind, and answer."""
        sources = {"person": person, "middle": middle, "kind": kind, "answer": answer}
        return {placeholder: sources[source] for placeholder, source in bindings}


MEMBERSHIP_AGENT_1_TEMPLATES = {
    "familiar": (
        ("p2g-f-00", "{person} belongs to the {group_kind} {group}."),
        ("p2g-f-01", "{person} is a member of the {group_kind} named {group}."),
        ("p2g-f-02", "The {group_kind} {group} counts {person} as a member."),
        ("p2g-f-03", "Among the members of the {group_kind} {group} is {person}."),
        ("p2g-f-04", "{person}'s affiliation is with the {group_kind} {group}."),
        ("p2g-f-05", "Membership records place {person} in the {group_kind} {group}."),
        ("p2g-f-06", "The {group_kind} to which {person} belongs is {group}."),
        ("p2g-f-07", "{group} is the {group_kind} that {person} joined."),
        ("p2g-f-08", "{person} serves as part of the {group_kind} {group}."),
        ("p2g-f-09", "The roster for the {group_kind} {group} includes {person}."),
        ("p2g-f-10", "{person} has membership in the {group_kind} called {group}."),
        ("p2g-f-11", "It is the {group_kind} {group} that has {person} as a member."),
        ("p2g-f-12", "There is a membership that places {person} in the {group_kind} {group}."),
        ("p2g-f-13", "The records state that {person} belongs to the {group_kind} {group}."),
        ("p2g-f-14", "What {person} belongs to is the {group_kind} {group}."),
        ("p2g-f-15", "In the {group_kind} {group}, one listed member is {person}."),
        ("p2g-f-16", "{person}, a member of the {group_kind} {group}, is carried on its roster."),
        ("p2g-f-17", "The member assigned to the {group_kind} {group} is {person}."),
        ("p2g-f-18", "Membership in the {group_kind} {group} is held by {person}."),
        ("p2g-f-19", "When the roster is checked, {person} is in the {group_kind} {group}."),
    ),
    "held_out": (
        ("p2g-h-00", "Asked about affiliation, {person} named the {group_kind} {group}."),
        ("p2g-h-01", "On the rolls of the {group_kind} {group}, one finds {person}."),
        ("p2g-h-02", "The membership of {person} lies with {group}, a {group_kind}."),
        ("p2g-h-03", "If you inspect the {group_kind} {group}, {person} appears on its roster."),
        ("p2g-h-04", "{person} and the {group_kind} {group} are linked by membership."),
        ("p2g-h-05", "A member associated with the {group_kind} {group} is {person}."),
        ("p2g-h-06", "For {person}, the listed {group_kind} is {group}."),
        ("p2g-h-07", "The affiliation entry beside {person}'s name reads: {group} {group_kind}."),
        ("p2g-h-08", "{person} was enrolled by the {group_kind} {group}."),
        ("p2g-h-09", "It appears on review that {person} is in the {group_kind} {group}."),
        ("p2g-h-10", "Enrolled in the {group_kind} {group}, {person} is a member."),
        ("p2g-h-11", "The {group_kind} known as {group} has enrolled {person}."),
        ("p2g-h-12", "As a member, {person} is attached to the {group_kind} {group}."),
        ("p2g-h-13", "The {group_kind} {group} is where {person} holds membership."),
        ("p2g-h-14", "Whoever checks the rolls will find {person} in the {group_kind} {group}."),
        ("p2g-h-15", "Under {group}, a {group_kind}, the member named is {person}."),
    ),
}

MEMBERSHIP_AGENT_2_TEMPLATES = {
    "familiar": (
        ("g2l-f-00", "The {group_kind} {group} is stationed in {location}."),
        ("g2l-f-01", "{location} is home to the {group_kind} {group}."),
        ("g2l-f-02", "The home base of the {group_kind} {group} is {location}."),
        ("g2l-f-03", "The {group_kind} named {group} operates from {location}."),
        ("g2l-f-04", "Operations for the {group_kind} {group} are based in {location}."),
        ("g2l-f-05", "You can find the {group_kind} {group} headquartered in {location}."),
        ("g2l-f-06", "{group}, which is a {group_kind}, has its base in {location}."),
        ("g2l-f-07", "From {location}, the {group_kind} {group} conducts its work."),
        ("g2l-f-08", "The base listed for the {group_kind} {group} is {location}."),
        ("g2l-f-09", "{location} hosts the headquarters of the {group_kind} {group}."),
        ("g2l-f-10", "The {group_kind} {group} calls {location} its home base."),
        ("g2l-f-11", "Headquarters records associate the {group_kind} {group} with {location}."),
        ("g2l-f-12", "There is a base for the {group_kind} {group} at {location}."),
        ("g2l-f-13", "The records state that the {group_kind} {group} works from {location}."),
        ("g2l-f-14", "Where the {group_kind} {group} is based is {location}."),
        ("g2l-f-15", "The {group_kind} {group} is assigned to the base at {location}."),
        ("g2l-f-16", "{location} is where the {group_kind} {group} keeps its headquarters."),
        ("g2l-f-17", "{location}, the base of the {group_kind} {group}, holds its headquarters."),
        ("g2l-f-18", "Keeping a base in {location} is what the {group_kind} {group} does."),
        ("g2l-f-19", "At {location}, one finds the base used by the {group_kind} {group}."),
    ),
    "held_out": (
        ("g2l-h-00", "To reach the {group_kind} {group}, one would travel to {location}."),
        ("g2l-h-01", "It is from {location} that the {group_kind} {group} works."),
        ("g2l-h-02", "For the {group_kind} {group}, headquarters can be found at {location}."),
        ("g2l-h-03", "A location tied to the {group_kind} {group} as its base is {location}."),
        ("g2l-h-04", "Where the {group_kind} {group} is based: {location}."),
        ("g2l-h-05", "{location} contains the operating base used by the {group_kind} {group}."),
        ("g2l-h-06", "The headquarters entry for {group}, a {group_kind}, names {location}."),
        ("g2l-h-07", "Based out of {location} is the {group_kind} known as {group}."),
        ("g2l-h-08", "The base at {location} was assigned to the {group_kind} {group}."),
        ("g2l-h-09", "It appears on review that the {group_kind} {group} works from {location}."),
        ("g2l-h-10", "Based in {location}, the {group_kind} {group} keeps its headquarters there."),
        ("g2l-h-11", "The {group_kind} known as {group} has its base at {location}."),
        ("g2l-h-12", "As its base, {location} is attached to the {group_kind} {group}."),
        ("g2l-h-13", "Headquarters for the {group_kind} {group} lie at {location}."),
        ("g2l-h-14", "Whoever seeks the {group_kind} {group} will find it in {location}."),
        ("g2l-h-15", "Under {group}, a {group_kind}, the listed base is {location}."),
    ),
}

MEMBERSHIP_QUESTION_TEMPLATES = {
    "familiar": (
        ("q-f-00", "Where is {person}'s group based?"),
        ("q-f-01", "In which place is the group containing {person} headquartered?"),
        ("q-f-02", "What is the home base of {person}'s group?"),
        ("q-f-03", "Where does the group that includes {person} operate from?"),
        ("q-f-04", "Which location hosts the group to which {person} belongs?"),
        ("q-f-05", "Name the place where {person}'s group is stationed."),
        ("q-f-06", "The group with {person} as a member is based where?"),
        ("q-f-07", "Where would one find the headquarters of {person}'s group?"),
        ("q-f-08", "For the group that includes {person}, name its home base."),
        ("q-f-09", "State the place where the group of {person} is based."),
        ("q-f-10", "It is which location that hosts the group of {person}?"),
        ("q-f-11", "At what place is the group including {person} based?"),
        ("q-f-12", "The group of {person} has its headquarters in which location?"),
        ("q-f-13", "Which place is the base that the group of {person} uses?"),
    ),
    "held_out": (
        ("q-h-00", "To visit the base of the group listing {person}, where should one go?"),
        ("q-h-01", "Identify the location associated with the group that counts {person} as a member."),
        ("q-h-02", "What place serves as headquarters for whichever group includes {person}?"),
        ("q-h-03", "Where is the operating base belonging to {person}'s affiliated group?"),
        ("q-h-04", "Which place is home to the group on whose roster {person} appears?"),
        ("q-h-05", "Give the headquarters location for the group affiliated with {person}."),
        ("q-h-06", "Which site is listed as the base for the group of {person}?"),
        ("q-h-07", "The group affiliated with {person} is quartered at which place?"),
        ("q-h-08", "Say where one would go to reach the group of {person}."),
        ("q-h-09", "If the group including {person} has a base, where is that base?"),
        ("q-h-10", "Whoever looks up {person} should find the group based in which place?"),
        ("q-h-11", "Under the group of {person}, which location is the listed base?"),
    ),
}

COLOR_AGENT_1_TEMPLATES = {
    "familiar": (
        ("col-a1-f-00", "{person}'s color is {color}."),
        ("col-a1-f-01", "The color assigned to {person} is {color}."),
        ("col-a1-f-02", "{person} is marked with the color {color}."),
        ("col-a1-f-03", "Records list {color} as the color for {person}."),
        ("col-a1-f-04", "What {person} is assigned is the color {color}."),
        ("col-a1-f-05", "There is a color, {color}, linked to {person}."),
        ("col-a1-f-06", "In the color ledger, {person} is entered as {color}."),
        ("col-a1-f-07", "{person}, whose color is {color}, is on the swatch list."),
    ),
    "held_out": (
        ("col-a1-h-00", "Asked for a color, {person} answered {color}."),
        ("col-a1-h-01", "The swatch beside {person}'s name is {color}."),
        ("col-a1-h-02", "It is {color} that was given to {person}."),
        ("col-a1-h-03", "{person} and the color {color} are paired."),
        ("col-a1-h-04", "For {person}, the noted color reads {color}."),
        ("col-a1-h-05", "Whoever looks up {person} finds the color {color}."),
        ("col-a1-h-06", "On the card for {person}, the color line says {color}."),
        ("col-a1-h-07", "Under {person}, the listed color is {color}."),
    ),
}

COLOR_AGENT_2_TEMPLATES = {
    "familiar": (
        ("col-a2-f-00", "The color {color} is the coat of the {animal}."),
        ("col-a2-f-01", "{animal} is the animal that wears {color}."),
        ("col-a2-f-02", "A coat of {color} belongs to the {animal}."),
        ("col-a2-f-03", "The animal wearing {color} is the {animal}."),
        ("col-a2-f-04", "Records state that {color} colors the {animal}."),
        ("col-a2-f-05", "Where {color} appears as a coat is on the {animal}."),
        ("col-a2-f-06", "The {animal} carries the color {color}."),
        ("col-a2-f-07", "Keeping a coat of {color} is what the {animal} does."),
    ),
    "held_out": (
        ("col-a2-h-00", "On review, {color} marks the {animal}."),
        ("col-a2-h-01", "The {animal} is recorded under the color {color}."),
        ("col-a2-h-02", "If the color is {color}, the animal is the {animal}."),
        ("col-a2-h-03", "As a coat color, {color} is attached to the {animal}."),
        ("col-a2-h-04", "Whoever seeks {color} will find the {animal}."),
        ("col-a2-h-05", "Under {color}, the listed animal is the {animal}."),
        ("col-a2-h-06", "The coat entry for {color} names the {animal}."),
        ("col-a2-h-07", "Based on {color}, the animal given is the {animal}."),
    ),
}

COLOR_QUESTION_TEMPLATES = {
    "familiar": (
        ("col-q-f-00", "Which animal has {person}'s color?"),
        ("col-q-f-01", "What animal wears the color assigned to {person}?"),
        ("col-q-f-02", "Name the animal whose coat is {person}'s color."),
        ("col-q-f-03", "The color linked to {person} belongs to which animal?"),
        ("col-q-f-04", "Which animal is marked with the color of {person}?"),
        ("col-q-f-05", "State the animal that carries {person}'s color."),
        ("col-q-f-06", "For the color of {person}, which animal wears it?"),
        ("col-q-f-07", "{person}'s color is the coat of which animal?"),
    ),
    "held_out": (
        ("col-q-h-00", "Which creature is listed under the color of {person}?"),
        ("col-q-h-01", "Identify the animal associated with {person}'s color."),
        ("col-q-h-02", "If {person} has a color, which animal wears it?"),
        ("col-q-h-03", "Whoever matches {person}'s color should name which animal?"),
        ("col-q-h-04", "The coat matching {person} belongs to which animal?"),
        ("col-q-h-05", "Say which animal carries the color given to {person}."),
        ("col-q-h-06", "Under the color of {person}, which animal is listed?"),
        ("col-q-h-07", "What animal does the swatch for {person} point to?"),
    ),
}

ANIMAL_AGENT_1_TEMPLATES = {
    "familiar": (
        ("ani-a1-f-00", "{person} looks after the {animal}."),
        ("ani-a1-f-01", "The creature in {person}'s care is the {animal}."),
        ("ani-a1-f-02", "There is an animal, the {animal}, kept by {person}."),
        ("ani-a1-f-03", "What {person} keeps is the {animal}."),
        ("ani-a1-f-04", "{person}, keeper of the {animal}, is on the list."),
        ("ani-a1-f-05", "Records place the {animal} with {person}."),
        ("ani-a1-f-06", "In the care ledger, {person} is entered with the {animal}."),
        ("ani-a1-f-07", "The {animal} is the creature assigned to {person}."),
    ),
    "held_out": (
        ("ani-a1-h-00", "Asked about a creature, {person} named the {animal}."),
        ("ani-a1-h-01", "The animal entry beside {person} reads {animal}."),
        ("ani-a1-h-02", "It is the {animal} that {person} tends."),
        ("ani-a1-h-03", "{person} and the {animal} are linked by care."),
        ("ani-a1-h-04", "For {person}, the listed creature is the {animal}."),
        ("ani-a1-h-05", "Whoever checks {person} will find the {animal}."),
        ("ani-a1-h-06", "On the card for {person}, the creature line says {animal}."),
        ("ani-a1-h-07", "Under {person}, the listed animal is the {animal}."),
    ),
}

ANIMAL_AGENT_2_TEMPLATES = {
    "familiar": (
        ("ani-a2-f-00", "The {animal} rests on the {furniture}."),
        ("ani-a2-f-01", "The furniture used by the {animal} is the {furniture}."),
        ("ani-a2-f-02", "A resting place for the {animal} is the {furniture}."),
        ("ani-a2-f-03", "Records state that the {animal} uses the {furniture}."),
        ("ani-a2-f-04", "Where the {animal} rests is the {furniture}."),
        ("ani-a2-f-05", "The {furniture} supports the {animal}."),
        ("ani-a2-f-06", "There is a piece of furniture, the {furniture}, for the {animal}."),
        ("ani-a2-f-07", "Keeping to the {furniture} is what the {animal} does."),
    ),
    "held_out": (
        ("ani-a2-h-00", "On review, the {animal} is set on the {furniture}."),
        ("ani-a2-h-01", "The {furniture} is recorded for the {animal}."),
        ("ani-a2-h-02", "If the creature is the {animal}, the furniture is the {furniture}."),
        ("ani-a2-h-03", "As a perch, the {furniture} is attached to the {animal}."),
        ("ani-a2-h-04", "Whoever seeks the {animal} will find the {furniture}."),
        ("ani-a2-h-05", "Under the {animal}, the listed furniture is the {furniture}."),
        ("ani-a2-h-06", "The rest entry for the {animal} names the {furniture}."),
        ("ani-a2-h-07", "Based on the {animal}, the furniture given is the {furniture}."),
    ),
}

ANIMAL_QUESTION_TEMPLATES = {
    "familiar": (
        ("ani-q-f-00", "Which furniture does {person}'s animal use?"),
        ("ani-q-f-01", "What piece of furniture belongs with the animal {person} keeps?"),
        ("ani-q-f-02", "Name the furniture used by {person}'s animal."),
        ("ani-q-f-03", "The animal linked to {person} rests on which furniture?"),
        ("ani-q-f-04", "Which piece supports the animal in {person}'s care?"),
        ("ani-q-f-05", "State the furniture where {person}'s animal rests."),
        ("ani-q-f-06", "For {person}'s animal, which furniture is assigned?"),
        ("ani-q-f-07", "{person}'s animal rests on which piece of furniture?"),
    ),
    "held_out": (
        ("ani-q-h-00", "Which furnishing is listed for the creature of {person}?"),
        ("ani-q-h-01", "Identify the furniture associated with {person}'s animal."),
        ("ani-q-h-02", "If {person} keeps an animal, which furniture does it use?"),
        ("ani-q-h-03", "Whoever matches {person}'s animal should name which furniture?"),
        ("ani-q-h-04", "The rest place for {person}'s creature is which piece?"),
        ("ani-q-h-05", "Say which furniture supports the animal tended by {person}."),
        ("ani-q-h-06", "Under the animal of {person}, which furniture is listed?"),
        ("ani-q-h-07", "What piece does the care record for {person} point to?"),
    ),
}

FURNITURE_AGENT_1_TEMPLATES = {
    "familiar": (
        ("fur-a1-f-00", "{person} owns the {furniture}."),
        ("fur-a1-f-01", "The piece of furniture belonging to {person} is the {furniture}."),
        ("fur-a1-f-02", "There is a piece, the {furniture}, owned by {person}."),
        ("fur-a1-f-03", "What {person} owns is the {furniture}."),
        ("fur-a1-f-04", "{person}, owner of the {furniture}, is on the inventory."),
        ("fur-a1-f-05", "Records place the {furniture} with {person}."),
        ("fur-a1-f-06", "In the inventory, {person} is entered with the {furniture}."),
        ("fur-a1-f-07", "The {furniture} is the piece assigned to {person}."),
    ),
    "held_out": (
        ("fur-a1-h-00", "Asked about furniture, {person} named the {furniture}."),
        ("fur-a1-h-01", "The inventory entry beside {person} reads {furniture}."),
        ("fur-a1-h-02", "It is the {furniture} that {person} holds."),
        ("fur-a1-h-03", "{person} and the {furniture} are linked by ownership."),
        ("fur-a1-h-04", "For {person}, the listed piece is the {furniture}."),
        ("fur-a1-h-05", "Whoever audits {person} will find the {furniture}."),
        ("fur-a1-h-06", "On the card for {person}, the furniture line says {furniture}."),
        ("fur-a1-h-07", "Under {person}, the listed piece is the {furniture}."),
    ),
}

FURNITURE_AGENT_2_TEMPLATES = {
    "familiar": (
        ("fur-a2-f-00", "The {furniture} stands in the {room}."),
        ("fur-a2-f-01", "The room that holds the {furniture} is the {room}."),
        ("fur-a2-f-02", "A place for the {furniture} is the {room}."),
        ("fur-a2-f-03", "Records state that the {furniture} is kept in the {room}."),
        ("fur-a2-f-04", "Where the {furniture} stands is the {room}."),
        ("fur-a2-f-05", "The {room} contains the {furniture}."),
        ("fur-a2-f-06", "There is a room, the {room}, for the {furniture}."),
        ("fur-a2-f-07", "Standing in the {room} is where the {furniture} belongs."),
    ),
    "held_out": (
        ("fur-a2-h-00", "On review, the {furniture} is placed in the {room}."),
        ("fur-a2-h-01", "The {room} is recorded for the {furniture}."),
        ("fur-a2-h-02", "If the piece is the {furniture}, the room is the {room}."),
        ("fur-a2-h-03", "As its room, the {room} is attached to the {furniture}."),
        ("fur-a2-h-04", "Whoever seeks the {furniture} will find the {room}."),
        ("fur-a2-h-05", "Under the {furniture}, the listed room is the {room}."),
        ("fur-a2-h-06", "The room entry for the {furniture} names the {room}."),
        ("fur-a2-h-07", "Based on the {furniture}, the room given is the {room}."),
    ),
}

FURNITURE_QUESTION_TEMPLATES = {
    "familiar": (
        ("fur-q-f-00", "In which room is {person}'s furniture?"),
        ("fur-q-f-01", "What room holds the piece owned by {person}?"),
        ("fur-q-f-02", "Name the room where {person}'s furniture stands."),
        ("fur-q-f-03", "The furniture linked to {person} is kept in which room?"),
        ("fur-q-f-04", "Which room contains the piece assigned to {person}?"),
        ("fur-q-f-05", "State the room that holds {person}'s furniture."),
        ("fur-q-f-06", "For {person}'s furniture, which room is listed?"),
        ("fur-q-f-07", "{person}'s piece of furniture stands in which room?"),
    ),
    "held_out": (
        ("fur-q-h-00", "Which chamber is listed for the furniture of {person}?"),
        ("fur-q-h-01", "Identify the room associated with {person}'s furniture."),
        ("fur-q-h-02", "If {person} owns a piece, in which room is it?"),
        ("fur-q-h-03", "Whoever matches {person}'s furniture should name which room?"),
        ("fur-q-h-04", "The place for {person}'s furnishing is which room?"),
        ("fur-q-h-05", "Say which room contains the piece held by {person}."),
        ("fur-q-h-06", "Under the furniture of {person}, which room is listed?"),
        ("fur-q-h-07", "What room does the inventory for {person} point to?"),
    ),
}

PLEDGED_GROUP_AGENT_1_TEMPLATES = {
    "familiar": (
        ("pld-a1-f-00", "{person} is pledged to the {group_kind} {group}."),
        ("pld-a1-f-01", "The {group_kind} that claims {person} is {group}."),
        ("pld-a1-f-02", "{person} sits with the {group_kind} named {group}."),
        ("pld-a1-f-03", "There is a pledge that binds {person} to the {group_kind} {group}."),
        ("pld-a1-f-04", "What {person} is pledged to is the {group_kind} {group}."),
        ("pld-a1-f-05", "{person}, pledged to the {group_kind} {group}, is on its roll."),
        ("pld-a1-f-06", "Records bind {person} to the {group_kind} {group}."),
        ("pld-a1-f-07", "In the pledge ledger, {person} is entered under the {group_kind} {group}."),
    ),
    "held_out": (
        ("pld-a1-h-00", "Asked about a pledge, {person} named the {group_kind} {group}."),
        ("pld-a1-h-01", "The pledge entry beside {person} reads {group} {group_kind}."),
        ("pld-a1-h-02", "It is the {group_kind} {group} that holds {person}'s pledge."),
        ("pld-a1-h-03", "{person} and the {group_kind} {group} are linked by a pledge."),
        ("pld-a1-h-04", "For {person}, the listed pledge is the {group_kind} {group}."),
        ("pld-a1-h-05", "Whoever reviews {person} will find the {group_kind} {group}."),
        ("pld-a1-h-06", "On the card for {person}, the pledge line says {group} {group_kind}."),
        ("pld-a1-h-07", "Under {person}, the listed {group_kind} pledge is {group}."),
    ),
}

PLEDGED_GROUP_AGENT_2_TEMPLATES = {
    "familiar": (
        ("pld-a2-f-00", "The {group_kind} {group} gathers in {location}."),
        ("pld-a2-f-01", "{location} is where the {group_kind} {group} convenes."),
        ("pld-a2-f-02", "A meeting place of the {group_kind} {group} is {location}."),
        ("pld-a2-f-03", "The hall assigned to the {group_kind} {group} is {location}."),
        ("pld-a2-f-04", "Records state that the {group_kind} {group} meets in {location}."),
        ("pld-a2-f-05", "Where the {group_kind} {group} gathers is {location}."),
        ("pld-a2-f-06", "The {group_kind} {group} calls {location} its meeting place."),
        ("pld-a2-f-07", "From {location}, the {group_kind} {group} holds its gathering."),
    ),
    "held_out": (
        ("pld-a2-h-00", "On review, the {group_kind} {group} convenes at {location}."),
        ("pld-a2-h-01", "The meeting entry for {group}, a {group_kind}, names {location}."),
        ("pld-a2-h-02", "If the pledge is the {group_kind} {group}, the hall is {location}."),
        ("pld-a2-h-03", "As its hall, {location} is attached to the {group_kind} {group}."),
        ("pld-a2-h-04", "Whoever seeks the {group_kind} {group} will find {location}."),
        ("pld-a2-h-05", "Under {group}, a {group_kind}, the listed hall is {location}."),
        ("pld-a2-h-06", "The hall entry for the {group_kind} {group} reads {location}."),
        ("pld-a2-h-07", "Based on the {group_kind} {group}, the meeting place given is {location}."),
    ),
}

PLEDGED_GROUP_QUESTION_TEMPLATES = {
    "familiar": (
        ("pld-q-f-00", "Where does the group pledged to {person} gather?"),
        ("pld-q-f-01", "In which place does {person}'s pledged group meet?"),
        ("pld-q-f-02", "What hall hosts the group {person} is pledged to?"),
        ("pld-q-f-03", "Name the place where {person}'s pledged group convenes."),
        ("pld-q-f-04", "The group that holds {person}'s pledge gathers where?"),
        ("pld-q-f-05", "Which location is the meeting place for {person}'s pledged group?"),
        ("pld-q-f-06", "State where the group bound to {person} gathers."),
        ("pld-q-f-07", "For {person}'s pledge, which place holds the meeting?"),
    ),
    "held_out": (
        ("pld-q-h-00", "Which hall is listed for the group pledged by {person}?"),
        ("pld-q-h-01", "Identify the meeting place of the group tied to {person}."),
        ("pld-q-h-02", "If {person} has a pledge, where does that group gather?"),
        ("pld-q-h-03", "Whoever matches {person}'s pledge should name which hall?"),
        ("pld-q-h-04", "The gathering place for {person}'s pledged group is which location?"),
        ("pld-q-h-05", "Say where the group holding {person}'s pledge convenes."),
        ("pld-q-h-06", "Under the pledge of {person}, which hall is listed?"),
        ("pld-q-h-07", "What meeting place does the pledge record for {person} point to?"),
    ),
}

GROUP_BINDINGS_AGENT_1 = (("person", "person"), ("group", "middle"), ("group_kind", "kind"))
GROUP_BINDINGS_AGENT_2 = (("group", "middle"), ("group_kind", "kind"), ("location", "answer"))

RELATION_DOMAINS = (
    RelationDomain(
        domain_id="membership",
        middle_source="groups",
        answer_source="locations",
        kinds=MEMBERSHIP_KINDS,
        agent_1_bindings=GROUP_BINDINGS_AGENT_1,
        agent_2_bindings=GROUP_BINDINGS_AGENT_2,
        agent_1_templates=MEMBERSHIP_AGENT_1_TEMPLATES,
        agent_2_templates=MEMBERSHIP_AGENT_2_TEMPLATES,
        question_templates=MEMBERSHIP_QUESTION_TEMPLATES,
    ),
    RelationDomain(
        domain_id="color",
        middle_source="colors",
        answer_source="coat_animals",
        kinds=(),
        agent_1_bindings=(("person", "person"), ("color", "middle")),
        agent_2_bindings=(("color", "middle"), ("animal", "answer")),
        agent_1_templates=COLOR_AGENT_1_TEMPLATES,
        agent_2_templates=COLOR_AGENT_2_TEMPLATES,
        question_templates=COLOR_QUESTION_TEMPLATES,
    ),
    RelationDomain(
        domain_id="animal",
        middle_source="kept_animals",
        answer_source="rest_furniture",
        kinds=(),
        agent_1_bindings=(("person", "person"), ("animal", "middle")),
        agent_2_bindings=(("animal", "middle"), ("furniture", "answer")),
        agent_1_templates=ANIMAL_AGENT_1_TEMPLATES,
        agent_2_templates=ANIMAL_AGENT_2_TEMPLATES,
        question_templates=ANIMAL_QUESTION_TEMPLATES,
    ),
    RelationDomain(
        domain_id="furniture",
        middle_source="owned_furniture",
        answer_source="rooms",
        kinds=(),
        agent_1_bindings=(("person", "person"), ("furniture", "middle")),
        agent_2_bindings=(("furniture", "middle"), ("room", "answer")),
        agent_1_templates=FURNITURE_AGENT_1_TEMPLATES,
        agent_2_templates=FURNITURE_AGENT_2_TEMPLATES,
        question_templates=FURNITURE_QUESTION_TEMPLATES,
    ),
    RelationDomain(
        domain_id="pledged_group",
        middle_source="groups",
        answer_source="locations",
        kinds=PLEDGED_GROUP_KINDS,
        agent_1_bindings=GROUP_BINDINGS_AGENT_1,
        agent_2_bindings=GROUP_BINDINGS_AGENT_2,
        agent_1_templates=PLEDGED_GROUP_AGENT_1_TEMPLATES,
        agent_2_templates=PLEDGED_GROUP_AGENT_2_TEMPLATES,
        question_templates=PLEDGED_GROUP_QUESTION_TEMPLATES,
    ),
)

DOMAINS_BY_ID = {domain.domain_id: domain for domain in RELATION_DOMAINS}
