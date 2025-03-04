"""
Literature Search Tool for AI Co-Scientist

This module provides tools for scientific literature search, citation management, and metadata extraction.
It integrates with multiple academic search engines and databases.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Data class for paper metadata."""
    
    title: str
    authors: List[str]
    abstract: str
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    citations: Optional[int] = None
    references: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    full_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "doi": self.doi,
            "journal": self.journal,
            "url": self.url,
            "citations": self.citations,
            "references": self.references,
            "keywords": self.keywords
        }
    
    def to_citation(self, format_type: str = "apa") -> str:
        """
        Generate a formatted citation.
        
        Args:
            format_type: Citation format ("apa", "mla", "chicago", "harvard", "bibtex")
            
        Returns:
            Formatted citation string
        """
        if format_type == "apa":
            # APA format
            author_text = ""
            if self.authors:
                if len(self.authors) == 1:
                    author_text = f"{self.authors[0]}."
                elif len(self.authors) == 2:
                    author_text = f"{self.authors[0]} & {self.authors[1]}."
                else:
                    author_text = f"{self.authors[0]} et al."
            
            year_text = f" ({self.year})." if self.year else ""
            journal_text = f" {self.journal}," if self.journal else ""
            doi_text = f" doi:{self.doi}" if self.doi else ""
            
            return f"{author_text}{year_text} {self.title}.{journal_text}{doi_text}"
            
        elif format_type == "bibtex":
            # BibTeX format
            first_author = self.authors[0].split(" ")[-1] if self.authors else "Unknown"
            year = self.year or "Unknown"
            key = f"{first_author}{year}"
            
            authors = " and ".join(self.authors) if self.authors else "Unknown"
            
            return (
                f"@article{{{key},\n"
                f"  author = {{{authors}}},\n"
                f"  title = {{{self.title}}},\n"
                f"  journal = {{{self.journal or 'Unknown'}}},\n"
                f"  year = {{{self.year or 'Unknown'}}},\n"
                f"  doi = {{{self.doi or ''}}}\n"
                f"}}"
            )
            
        # Default to a basic citation
        authors = ", ".join(self.authors) if self.authors else "Unknown"
        year = f"({self.year})" if self.year else ""
        journal = f"{self.journal}" if self.journal else ""
        
        return f"{authors} {year}. {self.title}. {journal}"


class CitationManager:
    """
    Manager for handling citations and bibliography.
    """
    
    def __init__(self):
        """Initialize the citation manager."""
        self.papers: Dict[str, PaperMetadata] = {}  # DOI -> PaperMetadata
        self.cached_search_results: Dict[str, List[PaperMetadata]] = {}
        
    def add_paper(self, paper: PaperMetadata) -> None:
        """
        Add a paper to the citation manager.
        
        Args:
            paper: Paper metadata to add
        """
        if paper.doi:
            self.papers[paper.doi] = paper
        else:
            # Use title as key if no DOI
            key = paper.title.lower().strip()
            existing = False
            
            # Check if we already have this paper
            for existing_paper in self.papers.values():
                if existing_paper.title.lower().strip() == key:
                    existing = True
                    break
                    
            if not existing:
                # Add with a generated key
                generated_key = f"paper_{len(self.papers)}"
                self.papers[generated_key] = paper
    
    def get_paper(self, identifier: str) -> Optional[PaperMetadata]:
        """
        Get a paper by DOI or title.
        
        Args:
            identifier: DOI or paper title
            
        Returns:
            Paper metadata if found, None otherwise
        """
        # Try as DOI first
        if identifier in self.papers:
            return self.papers[identifier]
            
        # Try as title
        for paper in self.papers.values():
            if paper.title.lower().strip() == identifier.lower().strip():
                return paper
                
        return None
    
    def generate_bibliography(self, format_type: str = "apa") -> str:
        """
        Generate a bibliography of all papers.
        
        Args:
            format_type: Citation format
            
        Returns:
            Formatted bibliography string
        """
        citations = []
        for paper in self.papers.values():
            citations.append(paper.to_citation(format_type))
            
        return "\n\n".join(citations)
    
    def export_library(self, file_path: str) -> None:
        """
        Export the citation library to a file.
        
        Args:
            file_path: Path to save the library
        """
        papers_dict = {
            k: v.to_dict() for k, v in self.papers.items()
        }
        
        with open(file_path, "w") as f:
            json.dump(papers_dict, f, indent=2)
    
    def import_library(self, file_path: str) -> None:
        """
        Import a citation library from a file.
        
        Args:
            file_path: Path to the library file
        """
        with open(file_path, "r") as f:
            papers_dict = json.load(f)
            
        for key, paper_dict in papers_dict.items():
            paper = PaperMetadata(
                title=paper_dict["title"],
                authors=paper_dict["authors"],
                abstract=paper_dict["abstract"],
                year=paper_dict.get("year"),
                doi=paper_dict.get("doi"),
                journal=paper_dict.get("journal"),
                url=paper_dict.get("url"),
                citations=paper_dict.get("citations"),
                references=paper_dict.get("references"),
                keywords=paper_dict.get("keywords")
            )
            self.papers[key] = paper
    
    def clear(self) -> None:
        """Clear all papers from the manager."""
        self.papers.clear()
        self.cached_search_results.clear()


class LiteratureSearch:
    """
    Tool for searching scientific literature across multiple sources.
    """
    
    def __init__(self, 
                email: str, 
                api_keys: Optional[Dict[str, str]] = None,
                citation_manager: Optional[CitationManager] = None):
        """
        Initialize the literature search tool.
        
        Args:
            email: Email for API access (required for PubMed)
            api_keys: Dictionary of API keys for different sources
            citation_manager: Citation manager to use
        """
        self.email = email
        self.api_keys = api_keys or {}
        self.citation_manager = citation_manager or CitationManager()
        
        # Default search parameters
        self.default_max_results = 10
        self.default_sort = "relevance"  # or "date"
        
        # Cache for search results
        self._cache = {}
        
    async def search_pubmed(self, 
                          query: str, 
                          max_results: int = 10, 
                          sort: str = "relevance",
                          full_text: bool = False,
                          **kwargs) -> List[PaperMetadata]:
        """
        Search PubMed for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order ("relevance" or "date")
            full_text: Whether to fetch full text when available
            
        Returns:
            List of paper metadata
        """
        # Build the cache key
        cache_key = f"pubmed:{query}:{max_results}:{sort}"
        if cache_key in self._cache:
            logger.info(f"Using cached results for PubMed query: {query}")
            return self._cache[cache_key]
            
        logger.info(f"Searching PubMed for: {query}")
        
        # PubMed API base URLs
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        search_url = f"{base_url}/esearch.fcgi"
        fetch_url = f"{base_url}/efetch.fcgi"
        
        # Search parameters
        sort_param = "relevance" if sort == "relevance" else "pub+date"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort_param,
            "retmode": "json",
            "email": self.email,
            "tool": "ai-co-scientist"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # First, search for matching PMIDs
                async with session.get(search_url, params=search_params) as response:
                    if response.status != 200:
                        logger.error(f"PubMed search error: {response.status}")
                        return []
                        
                    search_data = await response.json()
                    pmids = search_data.get("esearchresult", {}).get("idlist", [])
                    
                    if not pmids:
                        logger.info(f"No PubMed results found for query: {query}")
                        return []
                    
                    # Now fetch details for these PMIDs
                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(pmids),
                        "retmode": "xml",
                        "email": self.email,
                        "tool": "ai-co-scientist"
                    }
                    
                    async with session.get(fetch_url, params=fetch_params) as fetch_response:
                        if fetch_response.status != 200:
                            logger.error(f"PubMed fetch error: {fetch_response.status}")
                            return []
                            
                        xml_data = await fetch_response.text()
                        papers = self._parse_pubmed_xml(xml_data)
                        
                        # Cache the results
                        self._cache[cache_key] = papers
                        
                        # Add papers to citation manager
                        for paper in papers:
                            self.citation_manager.add_paper(paper)
                            
                        return papers
                        
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return []
    
    async def search_arxiv(self, 
                         query: str, 
                         max_results: int = 10, 
                         sort: str = "relevance",
                         categories: Optional[List[str]] = None,
                         **kwargs) -> List[PaperMetadata]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order ("relevance" or "date")
            categories: List of arXiv categories to search
            
        Returns:
            List of paper metadata
        """
        # Build the cache key
        cats_str = ",".join(categories) if categories else "all"
        cache_key = f"arxiv:{query}:{max_results}:{sort}:{cats_str}"
        if cache_key in self._cache:
            logger.info(f"Using cached results for arXiv query: {query}")
            return self._cache[cache_key]
            
        logger.info(f"Searching arXiv for: {query}")
        
        # arXiv API URL
        search_url = "http://export.arxiv.org/api/query"
        
        # Sort parameter
        sort_param = "relevance" if sort == "relevance" else "submittedDate"
        
        # Category filter
        cat_filter = ""
        if categories:
            cat_filter = " AND (" + " OR ".join([f"cat:{cat}" for cat in categories]) + ")"
        
        # Search parameters
        search_params = {
            "search_query": f"all:{query}{cat_filter}",
            "max_results": max_results,
            "sortBy": sort_param,
            "sortOrder": "descending"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params) as response:
                    if response.status != 200:
                        logger.error(f"arXiv search error: {response.status}")
                        return []
                        
                    xml_data = await response.text()
                    papers = self._parse_arxiv_xml(xml_data)
                    
                    # Cache the results
                    self._cache[cache_key] = papers
                    
                    # Add papers to citation manager
                    for paper in papers:
                        self.citation_manager.add_paper(paper)
                        
                    return papers
                    
        except Exception as e:
            logger.error(f"Error searching arXiv: {str(e)}")
            return []
    
    async def search_semantic_scholar(self, 
                                    query: str, 
                                    max_results: int = 10,
                                    full_text: bool = False,
                                    **kwargs) -> List[PaperMetadata]:
        """
        Search Semantic Scholar for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            full_text: Whether to fetch full text when available
            
        Returns:
            List of paper metadata
        """
        # Check if API key is available
        api_key = self.api_keys.get("semantic_scholar")
        if not api_key:
            logger.warning("No API key for Semantic Scholar, using limited access")
            
        # Build the cache key
        cache_key = f"semantic:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"Using cached results for Semantic Scholar query: {query}")
            return self._cache[cache_key]
            
        logger.info(f"Searching Semantic Scholar for: {query}")
        
        # Semantic Scholar API URL
        search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        # Search parameters
        search_params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,journal,url,citationCount,doi"
        }
        
        # Headers
        headers = {"x-api-key": api_key} if api_key else {}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=search_params, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Semantic Scholar search error: {response.status}")
                        return []
                        
                    search_data = await response.json()
                    papers = []
                    
                    for paper_data in search_data.get("data", []):
                        author_list = [author.get("name", "") for author in paper_data.get("authors", [])]
                        
                        paper = PaperMetadata(
                            title=paper_data.get("title", ""),
                            authors=author_list,
                            abstract=paper_data.get("abstract", ""),
                            year=paper_data.get("year"),
                            doi=paper_data.get("doi"),
                            journal=paper_data.get("journal", {}).get("name") if paper_data.get("journal") else None,
                            url=paper_data.get("url"),
                            citations=paper_data.get("citationCount")
                        )
                        papers.append(paper)
                    
                    # Fetch full text if requested and available
                    if full_text and api_key:
                        # This would require additional API calls to get full text
                        # Implementation depends on specific requirements
                        pass
                    
                    # Cache the results
                    self._cache[cache_key] = papers
                    
                    # Add papers to citation manager
                    for paper in papers:
                        self.citation_manager.add_paper(paper)
                        
                    return papers
                    
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {str(e)}")
            return []
    
    async def multi_source_search(self, 
                               query: str, 
                               sources: List[str] = None,
                               max_results: int = 10,
                               **kwargs) -> Dict[str, List[PaperMetadata]]:
        """
        Search multiple sources simultaneously.
        
        Args:
            query: Search query
            sources: List of sources to search
            max_results: Maximum results per source
            
        Returns:
            Dictionary mapping source names to result lists
        """
        if not sources:
            sources = ["pubmed", "arxiv", "semantic_scholar"]
            
        # Prepare search tasks
        tasks = []
        for source in sources:
            if source == "pubmed":
                tasks.append(self.search_pubmed(query, max_results, **kwargs))
            elif source == "arxiv":
                tasks.append(self.search_arxiv(query, max_results, **kwargs))
            elif source == "semantic_scholar":
                tasks.append(self.search_semantic_scholar(query, max_results, **kwargs))
                
        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        combined_results = {}
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.error(f"Error searching {source}: {str(result)}")
                combined_results[source] = []
            else:
                combined_results[source] = result
                
        return combined_results
    
    async def find_recent_papers(self, 
                             topic: str, 
                             max_age_years: int = 2,
                             max_results: int = 10) -> List[PaperMetadata]:
        """
        Find recent papers on a topic.
        
        Args:
            topic: Research topic
            max_age_years: Maximum age of papers in years
            max_results: Maximum number of results
            
        Returns:
            List of recent papers
        """
        current_year = datetime.now().year
        min_year = current_year - max_age_years
        
        # Search multiple sources
        all_results = await self.multi_source_search(
            query=topic,
            max_results=max_results * 2  # Get more results to filter
        )
        
        # Combine and filter results
        combined = []
        for source_results in all_results.values():
            combined.extend(source_results)
            
        # Filter by year
        recent_papers = [
            paper for paper in combined 
            if paper.year and int(paper.year) >= min_year
        ]
        
        # Sort by year (newest first) and limit results
        recent_papers.sort(key=lambda p: p.year or 0, reverse=True)
        return recent_papers[:max_results]
    
    async def find_highly_cited(self, 
                             topic: str, 
                             min_citations: int = 10,
                             max_results: int = 10) -> List[PaperMetadata]:
        """
        Find highly cited papers on a topic.
        
        Args:
            topic: Research topic
            min_citations: Minimum citation count
            max_results: Maximum number of results
            
        Returns:
            List of highly cited papers
        """
        # This mainly uses Semantic Scholar as it provides citation counts
        papers = await self.search_semantic_scholar(
            query=topic,
            max_results=max_results * 2  # Get more results to filter
        )
        
        # Filter by citation count
        cited_papers = [
            paper for paper in papers 
            if paper.citations and paper.citations >= min_citations
        ]
        
        # Sort by citation count (highest first) and limit results
        cited_papers.sort(key=lambda p: p.citations or 0, reverse=True)
        return cited_papers[:max_results]
    
    async def fetch_paper_details(self, doi: str) -> Optional[PaperMetadata]:
        """
        Fetch detailed information about a paper by DOI.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            Paper metadata if found, None otherwise
        """
        # Check if we already have this paper
        paper = self.citation_manager.get_paper(doi)
        if paper:
            return paper
            
        # Crossref API to resolve DOI
        crossref_url = f"https://api.crossref.org/works/{doi}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(crossref_url) as response:
                    if response.status != 200:
                        logger.error(f"Crossref API error: {response.status}")
                        return None
                        
                    data = await response.json()
                    paper_data = data.get("message", {})
                    
                    # Extract author names
                    authors = []
                    for author in paper_data.get("author", []):
                        given = author.get("given", "")
                        family = author.get("family", "")
                        authors.append(f"{given} {family}".strip())
                    
                    # Get publication year
                    published = paper_data.get("published-print", paper_data.get("published-online", {}))
                    year = published.get("date-parts", [[None]])[0][0]
                    
                    # Create paper metadata
                    paper = PaperMetadata(
                        title=paper_data.get("title", [""])[0],
                        authors=authors,
                        abstract="",  # Crossref doesn't usually include abstracts
                        year=year,
                        doi=doi,
                        journal=paper_data.get("container-title", [""])[0],
                        url=paper_data.get("URL")
                    )
                    
                    # Try to get abstract from other sources
                    await self._enrich_paper_data(paper)
                    
                    # Add to citation manager
                    self.citation_manager.add_paper(paper)
                    
                    return paper
                    
        except Exception as e:
            logger.error(f"Error fetching paper details: {str(e)}")
            return None
    
    async def _enrich_paper_data(self, paper: PaperMetadata) -> None:
        """
        Enrich paper data with information from other sources.
        
        Args:
            paper: Paper to enrich
        """
        # Try to get abstract from PubMed or Semantic Scholar
        if not paper.abstract and paper.doi:
            try:
                # Try Semantic Scholar first
                if "semantic_scholar" in self.api_keys:
                    ss_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper.doi}"
                    headers = {"x-api-key": self.api_keys["semantic_scholar"]}
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(ss_url, headers=headers) as response:
                            if response.status == 200:
                                data = await response.json()
                                if "abstract" in data and data["abstract"]:
                                    paper.abstract = data["abstract"]
                                    paper.citations = data.get("citationCount")
            except Exception as e:
                logger.warning(f"Error enriching paper data: {str(e)}")
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[PaperMetadata]:
        """
        Parse PubMed XML response to extract paper metadata.
        
        Args:
            xml_data: XML response from PubMed
            
        Returns:
            List of paper metadata
        """
        papers = []
        soup = BeautifulSoup(xml_data, "xml")
        
        for article in soup.find_all("PubmedArticle"):
            try:
                # Extract article data
                article_data = article.find("Article")
                if not article_data:
                    continue
                    
                # Title
                title = article_data.find("ArticleTitle")
                title_text = title.text if title else ""
                
                # Abstract
                abstract_elem = article_data.find("Abstract")
                abstract_text = ""
                if abstract_elem:
                    abstract_parts = abstract_elem.find_all("AbstractText")
                    if abstract_parts:
                        abstract_text = " ".join(part.text for part in abstract_parts)
                
                # Authors
                authors = []
                author_list = article_data.find("AuthorList")
                if author_list:
                    for author in author_list.find_all("Author"):
                        last_name = author.find("LastName")
                        fore_name = author.find("ForeName")
                        
                        if last_name and fore_name:
                            authors.append(f"{fore_name.text} {last_name.text}")
                        elif last_name:
                            authors.append(last_name.text)
                
                # Journal
                journal_elem = article_data.find("Journal")
                journal_name = ""
                if journal_elem:
                    journal_title = journal_elem.find("Title")
                    if journal_title:
                        journal_name = journal_title.text
                
                # Publication Date
                pub_date_elem = journal_elem.find("PubDate") if journal_elem else None
                year = None
                if pub_date_elem:
                    year_elem = pub_date_elem.find("Year")
                    if year_elem:
                        try:
                            year = int(year_elem.text)
                        except ValueError:
                            pass
                
                # DOI
                doi = None
                article_id_list = article.find("ArticleIdList")
                if article_id_list:
                    for article_id in article_id_list.find_all("ArticleId"):
                        if article_id.get("IdType") == "doi":
                            doi = article_id.text
                            break
                
                # Create paper metadata
                paper = PaperMetadata(
                    title=title_text,
                    authors=authors,
                    abstract=abstract_text,
                    year=year,
                    doi=doi,
                    journal=journal_name
                )
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error parsing PubMed article: {str(e)}")
        
        return papers
    
    def _parse_arxiv_xml(self, xml_data: str) -> List[PaperMetadata]:
        """
        Parse arXiv XML response to extract paper metadata.
        
        Args:
            xml_data: XML response from arXiv
            
        Returns:
            List of paper metadata
        """
        papers = []
        soup = BeautifulSoup(xml_data, "xml")
        
        for entry in soup.find_all("entry"):
            try:
                # Title
                title_elem = entry.find("title")
                title_text = title_elem.text.strip() if title_elem else ""
                
                # Abstract
                summary_elem = entry.find("summary")
                abstract_text = summary_elem.text.strip() if summary_elem else ""
                
                # Authors
                authors = []
                for author in entry.find_all("author"):
                    name_elem = author.find("name")
                    if name_elem:
                        authors.append(name_elem.text.strip())
                
                # Publication year
                published_elem = entry.find("published")
                year = None
                if published_elem:
                    try:
                        pub_date = published_elem.text.strip()
                        match = re.search(r"(\d{4})", pub_date)
                        if match:
                            year = int(match.group(1))
                    except ValueError:
                        pass
                
                # DOI and URL
                doi = None
                url = None
                for link in entry.find_all("link"):
                    href = link.get("href", "")
                    if link.get("title") == "doi":
                        doi = href.replace("http://dx.doi.org/", "")
                    elif link.get("rel") == "alternate":
                        url = href
                
                # Create paper metadata
                paper = PaperMetadata(
                    title=title_text,
                    authors=authors,
                    abstract=abstract_text,
                    year=year,
                    doi=doi,
                    journal="arXiv",
                    url=url
                )
                papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error parsing arXiv entry: {str(e)}")
        
        return papers
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._cache.clear()
        
    async def analyze_literature_trends(self, 
                                     topic: str, 
                                     years: int = 5) -> Dict[str, Any]:
        """
        Analyze trends in the literature for a given topic.
        
        Args:
            topic: Research topic
            years: Number of years to analyze
            
        Returns:
            Dictionary with trend analysis results
        """
        current_year = datetime.now().year
        start_year = current_year - years
        
        # Search for papers
        all_papers = []
        for year in range(start_year, current_year + 1):
            year_query = f"{topic} AND {year}[PDAT]" if "[PDAT]" not in topic else topic
            
            # Get papers for this year
            pubmed_papers = await self.search_pubmed(year_query, max_results=100)
            all_papers.extend(pubmed_papers)
            
        # Count papers per year
        papers_by_year = {}
        for paper in all_papers:
            if paper.year:
                year = int(paper.year)
                if year not in papers_by_year:
                    papers_by_year[year] = []
                papers_by_year[year].append(paper)
        
        # Extract common keywords
        keywords = self._extract_keywords(all_papers)
        
        # Identify top authors
        author_counts = {}
        for paper in all_papers:
            for author in paper.authors:
                if author not in author_counts:
                    author_counts[author] = 0
                author_counts[author] += 1
                
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Return trend analysis
        return {
            "topic": topic,
            "years_analyzed": years,
            "total_papers": len(all_papers),
            "papers_by_year": {year: len(papers) for year, papers in papers_by_year.items()},
            "top_keywords": keywords[:20],
            "top_authors": [{"name": name, "papers": count} for name, count in top_authors]
        }
    
    def _extract_keywords(self, papers: List[PaperMetadata]) -> List[Tuple[str, int]]:
        """
        Extract common keywords from paper abstracts.
        
        Args:
            papers: List of papers
            
        Returns:
            List of (keyword, count) tuples
        """
        # Combine all abstracts
        all_text = " ".join(paper.abstract for paper in papers if paper.abstract)
        
        # Simple keyword extraction (this could be enhanced with NLP libraries)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Remove common stop words
        stop_words = {"the", "and", "or", "a", "an", "in", "of", "to", "for", "with", 
                     "on", "by", "from", "at", "as", "that", "this", "these", "those", 
                     "is", "are", "was", "were", "has", "have", "had", "been"}
        filtered_words = [w for w in words if w not in stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
            
        # Sort by frequency
        return sorted(word_counts.items(), key=lambda x: x[1], reverse=True) 