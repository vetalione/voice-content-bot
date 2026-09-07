-- Optional cloud health checks: run once in your Supabase SQL Editor as postgres.
-- No bot token or API key is sent. This does not restore interrupted jobs or
-- protect against other Render restarts / a paused Supabase project.
create extension if not exists pg_cron with schema pg_catalog;
create extension if not exists pg_net with schema extensions;

-- A named schedule updates the existing job instead of adding duplicates.
select cron.schedule(
    'voice-content-render-health',
    '*/5 * * * *',
    $$select net.http_get(
        url := 'https://voice-content-bot.onrender.com/health',
        timeout_milliseconds := 45000
    );$$
);

-- Queue an initial check; requests execute after this transaction commits.
select net.http_get(
    url := 'https://voice-content-bot.onrender.com/health',
    timeout_milliseconds := 45000
);

-- To disable only this monitor later:
-- select cron.unschedule('voice-content-render-health');
-- To inspect recent HTTP results (cron success alone only means it queued a request):
-- select id, status_code, timed_out, error_msg, created
-- from net._http_response order by created desc limit 10;
