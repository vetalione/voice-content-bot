-- Run once in your own Supabase SQL editor. Backend service-role access only.
create table if not exists public.voice_checkpoints (
  job text not null,
  stage text not null,
  data jsonb not null,
  primary key (job, stage)
);
alter table public.voice_checkpoints enable row level security;
revoke all on public.voice_checkpoints from anon, authenticated;
grant all on public.voice_checkpoints to service_role;
