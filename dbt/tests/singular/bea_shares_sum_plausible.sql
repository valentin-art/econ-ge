-- share_it + share_nonit should not be materially above 1: a leak of rental
-- income into one bucket signals a bug in the omega (rental-share) split
-- upstream in src/features/bea/rental_prices.py.

select *
from {{ ref('bea_capital_services') }}
where share_it + share_nonit > 1.05
