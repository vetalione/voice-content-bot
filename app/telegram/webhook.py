"""Keep button subscriptions in sync without changing the webhook destination."""

import logging

logger = logging.getLogger(__name__)

ALLOWED_UPDATES = ["message", "channel_post", "edited_channel_post", "callback_query"]


async def sync_webhook_updates(bot, settings):
    if not settings.public_base_url:
        return
    try:
        info = await bot.get_webhook_info()
        expected = settings.public_base_url.rstrip("/") + settings.webhook_path
        if info.url != expected or info.has_custom_certificate:
            logger.warning(
                "Webhook subscription unchanged: destination/certificate requires manual setup"
            )
            return
        # An absent/empty allowed_updates list already includes callback_query.
        if not info.allowed_updates or "callback_query" in info.allowed_updates:
            return
        params = {
            "url": info.url,
            "allowed_updates": list(dict.fromkeys([*info.allowed_updates, *ALLOWED_UPDATES])),
            "max_connections": info.max_connections or 10,
            "drop_pending_updates": False,
        }
        if settings.webhook_secret:
            params["secret_token"] = settings.webhook_secret
        await bot.set_webhook(**params)
        logger.info("Telegram webhook subscription updated: callback_query enabled")
    except Exception as error:
        # Telegram transport errors may include a credential-bearing URL.
        logger.warning(
            "Webhook subscription sync failed: %s; run scripts/set_webhook.py set",
            type(error).__name__,
        )
