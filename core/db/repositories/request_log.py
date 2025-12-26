from typing import List, Optional

from sqlalchemy.orm import Session

from core.db import models


class RequestLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(
        self,
        *,
        input_type: str,
        model: Optional[str],
        status: str,
        error: Optional[str],
        duration_ms: Optional[float],
        text_len: Optional[int],
        token_count: Optional[int],
        image_width: Optional[int],
        image_height: Optional[int],
        response_preview: Optional[str],
    ) -> models.RequestLog:
        obj = models.RequestLog(
            input_type=input_type,
            model=model,
            status=status,
            error=error,
            duration_ms=duration_ms,
            text_len=text_len,
            token_count=token_count,
            image_width=image_width,
            image_height=image_height,
            response_preview=response_preview,
        )
        self.session.add(obj)
        return obj

    def list(self, limit: int = 200) -> List[models.RequestLog]:
        return (
            self.session.query(models.RequestLog)
            .order_by(models.RequestLog.id.desc())
            .limit(limit)
            .all()
        )

    def clear_all(self) -> int:
        count = self.session.query(models.RequestLog).delete()
        return count

    def stats(self):
        rows = self.session.query(models.RequestLog).all()
        count = len(rows)
        if count == 0:
            return {
                "count": 0,
                "duration_ms": {"mean": None, "p50": None, "p95": None, "p99": None},
                "text_len": {"mean": None, "p50": None},
                "token_count": None,
                "image_width": {"mean": None, "p50": None},
                "image_height": {"p50": None},
            }

        def _percentiles(values, qs):
            if not values:
                return {q: None for q in qs}
            vals = sorted(values)
            n = len(vals)
            res = {}
            for q in qs:
                if n == 1:
                    res[q] = vals[0]
                    continue
                k = (n - 1) * q
                f = int(k)
                c = min(f + 1, n - 1)
                if f == c:
                    res[q] = vals[int(k)]
                else:
                    res[q] = vals[f] * (c - k) + vals[c] * (k - f)
            return res

        durations = [r.duration_ms for r in rows if r.duration_ms is not None]
        text_lens = [r.text_len for r in rows if r.text_len is not None]
        token_counts = [r.token_count for r in rows if r.token_count is not None]
        widths = [r.image_width for r in rows if r.image_width is not None]
        heights = [r.image_height for r in rows if r.image_height is not None]

        dur_mean = sum(durations) / len(durations) if durations else None
        d_p = _percentiles(durations, [0.5, 0.95, 0.99]) if durations else {0.5: None, 0.95: None, 0.99: None}
        text_mean = sum(text_lens) / len(text_lens) if text_lens else None
        t_p = _percentiles(text_lens, [0.5]) if text_lens else {0.5: None}
        token_mean = sum(token_counts) / len(token_counts) if token_counts else None
        w_mean = sum(widths) / len(widths) if widths else None
        w_p = _percentiles(widths, [0.5]) if widths else {0.5: None}
        h_p = _percentiles(heights, [0.5]) if heights else {0.5: None}

        return {
            "count": count,
            "duration_ms": {"mean": dur_mean, "p50": d_p[0.5], "p95": d_p[0.95], "p99": d_p[0.99]},
            "text_len": {"mean": text_mean, "p50": t_p[0.5]},
            "token_count": token_mean,
            "image_width": {"mean": w_mean, "p50": w_p[0.5]},
            "image_height": {"p50": h_p[0.5]},
        }
