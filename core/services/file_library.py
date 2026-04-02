"""File Library Service for paginated document listing."""
from typing import List, Tuple, Optional, Callable


class FileLibraryService:
    """Service for managing paginated file library views."""

    def __init__(
        self,
        list_files_fn: Callable[[], List[str]],
        scan_dirs: Optional[List[str]] = None,
        page_size: int = 30,
    ):
        self.list_files_fn = list_files_fn
        self.scan_dirs = scan_dirs or []
        self.page_size = page_size

    def get_page(self, page: int) -> Tuple[List[str], Optional[int], Optional[int]]:
        """
        Get a page of file names.

        Returns:
            Tuple of (file_names, prev_page, next_page)
            prev_page is None if on first page
            next_page is None if on last page
        """
        files = self.list_files_fn()

        if not files:
            return [], None, None

        start = page * self.page_size
        end = start + self.page_size

        if start >= len(files):
            return [], None, None

        page_files = files[start:end]

        prev_page = page - 1 if page > 0 else None
        next_page = page + 1 if end < len(files) else None

        return page_files, prev_page, next_page
