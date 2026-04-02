"""Link Library Service for paginated link listing."""
from typing import List, Tuple, Optional, Callable


class LinkLibraryService:
    """Service for managing paginated link library views."""

    def __init__(
        self,
        list_links_fn: Callable[[int], List[Tuple[int, str]]],
        page_size: int = 10,
    ):
        self.list_links_fn = list_links_fn
        self.page_size = page_size

    def get_page(self, page: int) -> Tuple[List[Tuple[int, str]], Optional[int], Optional[int]]:
        """
        Get a page of links.

        Args:
            page: Page number (1-indexed)

        Returns:
            Tuple of (links, prev_page, next_page)
            prev_page is None if on first page
            next_page is None if on last page
        """
        links = self.list_links_fn(page)

        if not links:
            return [], None, None

        # Check if there might be more pages
        # This is a simplified check - real implementation would query total count
        has_more = len(links) == self.page_size

        prev_page = page - 1 if page > 1 else None
        next_page = page + 1 if has_more else None

        return links, prev_page, next_page
