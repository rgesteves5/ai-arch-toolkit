"""Research center agents — all built as generate_review_flow configurations."""

from __future__ import annotations

from ai_arch_toolkit.core._llm import LLM
from ai_arch_toolkit.core._tools import ToolGroup
from ai_arch_toolkit.nanope.research_center._wiki import (
    wiki_analysis_tools,
    wiki_notes_tools,
    wiki_read_tools,
    wiki_write_tools,
)
from ai_arch_toolkit.nanope.research_center.reasoning import make_reasoning_tool
from ai_arch_toolkit.toolkit.agents.flows._generate_review import generate_review_flow
from ai_arch_toolkit.toolkit.flow._flow import Flow
from ai_arch_toolkit.toolkit.memory.graph._store import GraphStore
from ai_arch_toolkit.toolkit.tools._dictionary import define_word
from ai_arch_toolkit.toolkit.tools._web import http_get, scrape_text
from ai_arch_toolkit.toolkit.tools._wikipedia import (
    wikipedia_article,
    wikipedia_related,
    wikipedia_search,
)


def researcher_agent(
    llm: LLM,
    wiki: GraphStore,
    *,
    max_cycles: int = 2,
    max_gen_iterations: int = 8,
    max_review_iterations: int = 3,
) -> Flow:
    """Researcher — gathers knowledge from external sources and stores it in the wiki.

    Generate phase: searches wikipedia/dictionary, stores findings in wiki.
    Review phase: checks if research is thorough enough, retries if gaps remain.
    """
    write_tools = wiki_write_tools(wiki)
    read_tools = wiki_read_tools(wiki)

    gen_tools = ToolGroup(
        wikipedia_search,
        wikipedia_article,
        wikipedia_related,
        define_word,
        *write_tools.tools,
        *read_tools.tools,
    )
    review_tools = ToolGroup(*read_tools.tools)

    return generate_review_flow(
        gen_llm=llm,
        review_llm=llm,
        gen_tools=gen_tools,
        review_tools=review_tools,
        gen_system=(
            "You are a research agent. Your job is to thoroughly research the given topic "
            "using wikipedia and dictionary tools, then store your findings in the wiki.\n\n"
            "IMPORTANT: Before storing any knowledge, you MUST:\n"
            "1. Call wiki_find_duplicates with the subject to check for existing entries.\n"
            "2. Only create a new entry if the content is genuinely distinct.\n\n"
            "For each distinct piece of knowledge you find:\n"
            "1. Use wiki_remember with structured fields:\n"
            "   - title: Short descriptive title\n"
            "   - summary: Concise 1-2 sentence summary\n"
            "   - category: Use a fitting category — fact, concept, definition, event,\n"
            "     biography, law, theory, method, or any other fitting category\n"
            "   - subject: The main entity or topic\n"
            "   - details: Additional details, examples, elaboration\n"
            "2. Use wiki_connect to link related entries.\n"
            "3. Set the subject field to the main entity.\n\n"
            "Be thorough — search for the main topic and related subtopics. "
            "Store structured, atomic facts rather than long paragraphs."
        ),
        review_system=(
            "You are a research reviewer. Evaluate if the research is thorough.\n\n"
            "Use wiki_search and wiki_categories to check what was stored.\n"
            "Consider:\n"
            "- Are the key aspects of the topic covered?\n"
            "- Are there obvious subtopics that were missed?\n"
            "- Are the entries well-categorized and connected?\n"
            "- Are entries using diverse categories (not just 'fact')?\n\n"
            "Respond with ACCEPT if the research is satisfactory.\n"
            "Respond with RETRY followed by specific gaps to fill."
        ),
        max_cycles=max_cycles,
        max_gen_iterations=max_gen_iterations,
        max_review_iterations=max_review_iterations,
    )


def linker_agent(
    llm: LLM,
    wiki: GraphStore,
    *,
    max_cycles: int = 2,
    max_gen_iterations: int = 6,
    max_review_iterations: int = 3,
) -> Flow:
    """Linker — analyzes the wiki graph and creates meaningful connections.

    Generate phase: explores the graph, finds orphan nodes, proposes and creates links.
    Review phase: validates that connections are meaningful.
    """
    read_tools = wiki_read_tools(wiki)
    write_tools = wiki_write_tools(wiki)

    gen_tools = ToolGroup(*read_tools.tools, *write_tools.tools)
    review_tools = ToolGroup(*read_tools.tools)

    return generate_review_flow(
        gen_llm=llm,
        review_llm=llm,
        gen_tools=gen_tools,
        review_tools=review_tools,
        gen_system=(
            "You are a knowledge linker. Your job is to analyze the wiki and create "
            "meaningful connections between knowledge nodes.\n\n"
            "PRIORITY ORDER:\n"
            "1. Connect orphan nodes first — these have no connections at all.\n"
            "2. Create cross-category links (e.g. connect a biography to a theory).\n"
            "3. Prefer specific relations over generic 'related_to'.\n\n"
            "Steps:\n"
            "1. Use wiki_categories to see what's in the wiki.\n"
            "2. Use wiki_search with broad terms to find nodes.\n"
            "3. Use wiki_read to understand node content.\n"
            "4. Use wiki_connect to link related nodes with specific relations:\n"
            "   - related_to: general topical relation\n"
            "   - subtopic_of: X is a narrower topic within Y\n"
            "   - supports: X provides evidence for Y\n"
            "   - contradicts: X conflicts with Y\n"
            "   - defines: X is a definition relevant to Y\n"
            "   - discovered_by: X was discovered/created by Y\n"
            "   - precedes / follows: chronological ordering\n"
            "   - generalizes / specializes: abstraction hierarchy\n"
            "   - derived_from: X is derived or adapted from Y\n"
            "   - applied_to: X is applied in the context of Y\n"
            "   - part_of: X is a component or part of Y\n"
            "   - causes / enables: causal relations\n"
            "   - Or create new relation types as needed.\n\n"
            "Focus on finding non-obvious connections across different categories."
        ),
        review_system=(
            "You are a link reviewer. Check if the connections made are meaningful.\n\n"
            "Use wiki_explore to verify the connections.\n"
            "Consider:\n"
            "- Are the relation types appropriate and specific?\n"
            "- Are there orphan nodes that should be connected?\n"
            "- Are there redundant or incorrect links?\n"
            "- Are too many links just 'related_to' instead of specific relations?\n\n"
            "Respond with ACCEPT if the linking is satisfactory.\n"
            "Respond with RETRY followed by suggestions for better links."
        ),
        max_cycles=max_cycles,
        max_gen_iterations=max_gen_iterations,
        max_review_iterations=max_review_iterations,
    )


def manager_agent(
    llm: LLM,
    wiki: GraphStore,
    *,
    project_id: str = "",
    max_cycles: int = 1,
    max_gen_iterations: int = 6,
    max_review_iterations: int = 3,
) -> Flow:
    """Manager — reviews overall research quality, identifies gaps, directs next steps.

    Generate phase: analyzes wiki state, web-checks coverage, decides next action.
    Review phase: validates the assessment is fair and actionable.

    Uses read-only wiki tools, web tools, and reasoning strategies.
    """
    read_tools = wiki_read_tools(wiki)
    reasoning = make_reasoning_tool()

    gen_tool_list = [
        *read_tools.tools,
        http_get,
        scrape_text,
        reasoning,
    ]
    review_tool_list = [*read_tools.tools, reasoning]

    # Add notes tools if project_id is available
    if project_id:
        notes_tools = wiki_notes_tools(project_id)
        gen_tool_list.extend(notes_tools.tools)
        review_tool_list.extend(notes_tools.tools)

    gen_tools = ToolGroup(*gen_tool_list)
    review_tools = ToolGroup(*review_tool_list)

    notes_instructions = ""
    if project_id:
        notes_instructions = (
            "\n\nNOTES: You have persistent notes that survive across pipeline cycles.\n"
            "Use notes_write to save scratch notes, ideas, TODOs, or strategy decisions.\n"
            "Use notes_read to recall your previous notes.\n"
            "This helps you maintain continuity across research cycles.\n"
        )

    return generate_review_flow(
        gen_llm=llm,
        review_llm=llm,
        gen_tools=gen_tools,
        review_tools=review_tools,
        gen_system=(
            "You are the research manager. You report to the Owner and direct the\n"
            "research team (Researcher, Linker, Reviewer, Writer).\n\n"
            "The task you receive contains the Owner's instructions — their research\n"
            "topic, desired report format, focus areas, audience, and any constraints.\n"
            "You must honour the Owner's brief and translate it into actionable plans\n"
            "for each team member.\n\n"
            "Steps:\n"
            "1. Read the Owner's instructions carefully.\n"
            "2. Use wiki_categories to see the current state of the knowledge base.\n"
            "3. Use wiki_search to sample entries and assess quality.\n"
            "4. Use reasoning_strategy tool (try 'gap_analysis' or 'topic_decomposition')\n"
            "   to structure your assessment.\n"
            "5. Optionally use http_get or scrape_text to check if important\n"
            "   aspects are missing from the wiki.\n"
            "6. Browse available reasoning strategies (call reasoning_strategy with no args)\n"
            "   and choose the best one for the Writer to use.\n"
            f"{notes_instructions}\n"
            "Your output MUST follow this exact structure:\n\n"
            "COVERAGE: <percentage estimate of topic coverage relative to Owner's brief>\n\n"
            "DECISION: <one of: RESEARCH_MORE | READY_TO_WRITE | DONE>\n\n"
            "RESEARCHER_PLAN:\n"
            "<Specific subtopics to research next. List concrete search queries.\n"
            " Align with the Owner's focus areas and priorities.\n"
            " If READY_TO_WRITE or DONE, write 'N/A'.>\n\n"
            "LINKER_PLAN:\n"
            "<Instructions for the linker: what types of connections to focus on,\n"
            " which categories need better linking, etc.\n"
            " If READY_TO_WRITE or DONE, write 'N/A'.>\n\n"
            "REVIEWER_PLAN:\n"
            "<Instructions for the graph reviewer: what quality issues to look for,\n"
            " acceptable orphan percentage, expected category diversity, etc.\n"
            " If READY_TO_WRITE or DONE, write 'N/A'.>\n\n"
            "WRITER_PLAN:\n"
            "<Instructions for the writer: recommended structure, themes to emphasize,\n"
            " target audience, tone, and report format — all guided by the Owner's brief.>\n\n"
            "WRITER_STRATEGY: <name of a reasoning strategy from the catalog for\n"
            " the writer to use, e.g. 'synthesis' or 'compare_contrast'>\n"
        ),
        review_system=(
            "You are a manager reviewer. Evaluate if the plan is actionable and\n"
            "aligned with the Owner's instructions.\n\n"
            "Use wiki_search and wiki_categories to verify coverage claims.\n"
            "Use reasoning_strategy tool if helpful.\n"
            "Check that:\n"
            "- The plan respects the Owner's requested focus, format, and audience.\n"
            "- The DECISION matches the stated COVERAGE.\n"
            "- RESEARCHER_PLAN has concrete, searchable topics (not vague).\n"
            "- LINKER_PLAN identifies specific connection opportunities.\n"
            "- REVIEWER_PLAN gives clear quality targets.\n"
            "- WRITER_PLAN gives clear structural guidance matching the Owner's brief.\n"
            "- WRITER_STRATEGY names a valid reasoning strategy.\n\n"
            "Respond with ACCEPT if the plan is accurate and actionable.\n"
            "Respond with RETRY followed by corrections."
        ),
        max_cycles=max_cycles,
        max_gen_iterations=max_gen_iterations,
        max_review_iterations=max_review_iterations,
    )


def review_agent(
    llm: LLM,
    wiki: GraphStore,
    *,
    max_cycles: int = 1,
    max_gen_iterations: int = 6,
    max_review_iterations: int = 3,
) -> Flow:
    """Reviewer — analyzes wiki graph quality and produces a quality report.

    Generate phase: runs graph stats, finds orphans, checks duplicates.
    Review phase: validates the assessment is accurate.
    """
    read_tools = wiki_read_tools(wiki)
    analysis_tools = wiki_analysis_tools(wiki)

    gen_tools = ToolGroup(*read_tools.tools, *analysis_tools.tools)
    review_tools = ToolGroup(*read_tools.tools)

    return generate_review_flow(
        gen_llm=llm,
        review_llm=llm,
        gen_tools=gen_tools,
        review_tools=review_tools,
        gen_system=(
            "You are a wiki graph quality reviewer. Analyze the wiki knowledge graph\n"
            "and produce a detailed quality report.\n\n"
            "Steps:\n"
            "1. Call wiki_graph_stats to get overall graph statistics.\n"
            "2. Call wiki_find_orphans to identify unconnected nodes.\n"
            "3. Use wiki_search and wiki_read to spot-check node quality.\n"
            "4. Look for potential duplicates by searching common subjects.\n\n"
            "Your output MUST follow this exact structure:\n\n"
            "GRAPH_STATS:\n"
            "<Total nodes, edges, orphan count and percentage>\n\n"
            "CATEGORY_DISTRIBUTION:\n"
            "<List each category with count and percentage>\n\n"
            "EDGE_RELATION_DISTRIBUTION:\n"
            "<List each relation type with count>\n\n"
            "DUPLICATE_CANDIDATES:\n"
            "<List any nodes that appear to be duplicates, or 'None found'>\n\n"
            "TOP_ISSUES:\n"
            "<Numbered list of the most important quality issues>\n\n"
            "RECOMMENDATIONS:\n"
            "<Specific actions to improve graph quality>\n\n"
            "QUALITY_SCORE: <1-10 rating of overall graph quality>\n"
        ),
        review_system=(
            "You are a review checker. Verify the quality report is accurate.\n\n"
            "Use wiki_search and wiki_read to spot-check the claims.\n"
            "Check that:\n"
            "- Statistics match what you can verify.\n"
            "- Duplicate candidates are genuine potential duplicates.\n"
            "- Issues and recommendations are actionable.\n"
            "- Quality score is reasonable given the findings.\n\n"
            "Respond with ACCEPT if the report is accurate.\n"
            "Respond with RETRY followed by corrections."
        ),
        max_cycles=max_cycles,
        max_gen_iterations=max_gen_iterations,
        max_review_iterations=max_review_iterations,
    )


def writer_agent(
    llm: LLM,
    wiki: GraphStore,
    *,
    max_cycles: int = 2,
    max_gen_iterations: int = 6,
    max_review_iterations: int = 4,
) -> Flow:
    """Writer — synthesizes wiki knowledge into a structured report.

    Generate phase: reads wiki content, uses reasoning strategies to structure output.
    Review phase: checks coherence, completeness, and grounding in wiki sources.
    """
    read_tools = wiki_read_tools(wiki)
    reasoning = make_reasoning_tool()

    gen_tools = ToolGroup(*read_tools.tools, reasoning)
    review_tools = ToolGroup(*read_tools.tools, reasoning)

    return generate_review_flow(
        gen_llm=llm,
        review_llm=llm,
        gen_tools=gen_tools,
        review_tools=review_tools,
        gen_system=(
            "You are a research writer. Your job is to synthesize the knowledge in the "
            "wiki into a well-structured report.\n\n"
            "Steps:\n"
            "1. Use wiki_categories to understand what's available.\n"
            "2. Use wiki_search to find all relevant entries for the topic.\n"
            "3. Use wiki_read to get full details of key nodes.\n"
            "4. Use reasoning_strategy (try 'synthesis' or 'compare_contrast') to plan "
            "your report structure.\n\n"
            "Your report should:\n"
            "- Have a clear title and introduction.\n"
            "- Be organized by themes/subtopics, not by source.\n"
            "- Include specific facts and details from the wiki.\n"
            "- Note any uncertainties or gaps in the research.\n"
            "- Be comprehensive but concise."
        ),
        review_system=(
            "You are a writing reviewer. Evaluate the report quality.\n\n"
            "Use wiki_search to verify claims are grounded in the wiki.\n"
            "Use reasoning_strategy if helpful.\n"
            "Consider:\n"
            "- Is the report well-structured and readable?\n"
            "- Does it cover the key topics from the wiki?\n"
            "- Are there unsupported claims or missing important facts?\n\n"
            "Respond with ACCEPT if the report is satisfactory.\n"
            "Respond with RETRY followed by specific improvements needed."
        ),
        max_cycles=max_cycles,
        max_gen_iterations=max_gen_iterations,
        max_review_iterations=max_review_iterations,
    )
